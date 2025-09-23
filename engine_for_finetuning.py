import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform


#def visualize_embeddings(embeddings_list, labels_list):
def visualize_embeddings(embeddings_list, labels_list, save_path=None):

    if save_path is None:
        save_path = '/data/videomae_outputs/test_embeddings/embeddings_tsne.png'
    # > combine all 
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"\n[Embedding Visualization]")
    print(f"  Total samples: {len(embeddings)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    if len(embeddings) <= 3:
        print("  Too few samples for visualization")
        return
    
    # > PCA 50dims
    #if embeddings.shape[1] > 50:
    if embeddings.shape[1] > 50 and len(embeddings) > 50:
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_[:5]}")
    elif len(embeddings) < embeddings.shape[1]:
        # > When the number of samples is less than the feature dimension, use the number of samples minus 1 as the number of components
        pca = PCA(n_components=min(len(embeddings)-1, 50))
        embeddings_pca = pca.fit_transform(embeddings)
    else:
        embeddings_pca = embeddings
    
    # > t-SNE 2D (only when num_sample > 3)
    #tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    #embeddings_2d = tsne.fit_transform(embeddings_pca)
    if len(embeddings) > 3:
        perplexity = min(5, len(embeddings)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # > plot
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=f'Class {label}', alpha=0.6, s=50)
    
    plt.legend()
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved to {save_path}")

def analyze_embedding_quality(embeddings, labels):
    
    # > silhouette score
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(embeddings, labels)
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
    
    # > calculate the distance within the class and between classes
    distances = squareform(pdist(embeddings, 'euclidean'))
    
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() > 0:  # make sure the class is not empty
            intra_distances = distances[mask][:, mask]
            if intra_distances.shape[0] > 1:
                intra_class = intra_distances[np.triu_indices_from(intra_distances, k=1)].mean()
            else:
                intra_class = 0
            
            if (~mask).sum() > 0:  # make sure other classes are not empty
                inter_class = distances[mask][:, ~mask].mean()
                ratio = inter_class/intra_class if intra_class > 0 else float('inf')
                print(f"  Class {label}: Intra-dist={intra_class:.4f}, Inter-dist={inter_class:.4f}, Ratio={ratio:.4f}")


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)

    print(f"Loss value: {loss.item()}")
    if torch.isnan(loss):
        print("Loss is NaN!")
        print(f"Outputs: {outputs}")
        print(f"Targets: {target}")

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # > debug: visualize and analyze the embeddings during training
    collect_embeddings = (epoch % 1 == 0)  # > every epoch
    if collect_embeddings:
        train_embeddings = []
        train_labels = []


    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            break #continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            #with torch.cuda.amp.autocast():
            loss, output = train_class_batch(
                model, samples, targets, criterion)

        loss_value = loss.item()

        # > debug: get the features from training
        if data_iter_step == 0: 
            with torch.no_grad():
                # > get features
                samples_float = samples.float() if samples.dtype == torch.float16 else samples
                if hasattr(model, 'module'):  # DistributedDataParallel
                    features = model.module.forward_features(samples_float)
                else:
                    features = model.forward_features(samples_float)
                # Feature collapse metrics
                import torch.nn.functional as F
                feature_std = features.std(dim=0).mean()
                feat_norm = F.normalize(features, p=2, dim=1)
                similarity_matrix = torch.mm(feat_norm, feat_norm.t())
                avg_similarity = similarity_matrix[~torch.eye(len(features), dtype=bool, device=features.device)].mean()

                print(f"\n[FEATURE COLLAPSE CHECK - Step {data_iter_step}]")
                print(f"  Feature std: {feature_std:.6f} (healthy >0.1, collapsed <0.01)")
                print(f"  Avg pairwise similarity: {avg_similarity:.4f} (>0.95 = collapsed)")
                print(f"  Unique predictions: {len(output.argmax(dim=1).unique())}/{len(output)}")

            print(f"\n[EPOCH {epoch} DEBUG]")
            print(f"  Batch shape: {samples.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output values: {output[:4].detach().cpu()}")
            print(f"  Predictions: {output.argmax(dim=1)[:4].detach().cpu()}")

            # > in case there's mixup
            if mixup_fn is None:
                print(f"  True labels: {targets[:4].cpu()}")
            else:
                print(f"  Mixed targets shape: {targets.shape}")

            print(f"  Loss: {loss_value:.4f}")
            print(f"  Feature stats: mean={features.mean().item():.4f}, std={features.std().item():.4f}")

            # > check output difference
            if len(output) > 1:
                output_diff = (output[0] - output[1:]).abs().max().item()
                print(f"  Max output difference: {output_diff:.6f}")
                if output_diff < 1e-5:
                    print("  ⚠️ WARNING: All outputs are nearly identical!")

            # > check gradient after back propogation
            if data_iter_step == 0 and epoch == 0:
                print("\n[Gradient Check]")
                for name, param in model.named_parameters():
                    if param.requires_grad and "head" in name:
                        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

        # > collect embeddings
        # if collect_embeddings and data_iter_step % 5 == 0:  # every 5 bathces FOR LARGER DATASETS
        if collect_embeddings and data_iter_step % 1 == 0: # every batch FOR SMALLER DATASET
            with torch.no_grad():
                samples_float = samples.float() if samples.dtype == torch.float16 else samples
                if hasattr(model, 'module'):
                    features = model.module.forward_features(samples_float)
                else:
                    features = model.forward_features(samples_float)
                
                train_embeddings.append(features.cpu().numpy())
                if mixup_fn is None:
                    train_labels.append(targets.cpu().numpy())
                else:
                    # when mixed up, use the argmax of the original label
                    train_labels.append(targets.argmax(dim=1).cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if torch.isnan(loss):
            print(f"Loss is nan before backward")
            print(f"Output values: {output}")
            print(f"Target values: {targets}")

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if grad_norm is None or (isinstance(grad_norm, float) and math.isnan(grad_norm)):
                print(f"grad_norm is NaN or None: {grad_norm}")
                print(f"Loss scale: {loss_scaler.state_dict()['scale']}")

                if grad_norm is None:
                    print(f"[GRAD NONE DEBUG]")
                    print(f"  Grad skipped. Loss scale: {loss_scaler.state_dict()['scale']}")
                    print(f"  Loss value: {loss_value}")
                    print(f"  Is loss finite: {torch.isfinite(torch.tensor(loss_value)).item()}")
                    print(f"  Update freq: {update_freq}, data_iter_step: {data_iter_step}")
                    print(f"  Should update: {(data_iter_step + 1) % update_freq == 0}")

                    if not hasattr(train_one_epoch, 'grad_none_count'):
                        train_one_epoch.grad_none_count = 0
                    train_one_epoch.grad_none_count += 1
                    print(f"  Total grad_none occurrences: {train_one_epoch.grad_none_count}")

            # > which layer yielded grad_norm = nan
            if (data_iter_step + 1) % update_freq == 0:  # > check only when it's updated
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in {name}")
                            print(f"  Param stats: mean={param.mean():.6f}, std={param.std():.6f}")
                            print(f"  Grad max: {param.grad.max()}, min: {param.grad.min()}")
                            # > print first few grads
                            print(f"  First few grads: {param.grad.flatten()[:10]}")

            # > debug: check gradient after back propogation
            if data_iter_step == 0 and (data_iter_step + 1) % update_freq == 0:
                print("\n[Gradient Analysis - Epoch {}]".format(epoch))
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm_val = param.grad.data.norm(2).item()
                        if grad_norm_val == 0:
                            print(f"Zero gradient: {name}")
                        elif grad_norm_val > 100:
                            print(f"Large gradient: {name} = {grad_norm_val:.2f}")
                        elif "head" in name or "fc_norm" in name: # print the grad_norm of important layers
                            print(f"  {name}: grad_norm = {grad_norm_val:.4f}")

                # > get total norm
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        param_count += 1
                total_norm = total_norm ** 0.5
                print(f"  Total gradient norm: {total_norm:.4f}")
                print(f"  Parameters with gradients: {param_count}")

                # > dead layer check
                dead_layers = []
                zero_grad_layers = []
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_magnitude = param.grad.abs().mean().item()
                        if grad_magnitude < 1e-8:
                            dead_layers.append(name)
                        if grad_magnitude == 0:
                            zero_grad_layers.append(name)

                if dead_layers:
                    print(f"\n[DEAD LAYERS]: {len(dead_layers)}/{sum(p.requires_grad for p in model.parameters())}")
                    print(f"  First 5: {dead_layers[:5]}")
                if zero_grad_layers:
                    print(f"  Zero gradient layers: {len(zero_grad_layers)}")


            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


    
    if collect_embeddings and train_embeddings:
        import numpy as np
        all_features = np.concatenate(train_embeddings)
        all_labels = np.concatenate(train_labels)

        print(f"\n[Training Embeddings Analysis - Epoch {epoch}]")
        analyze_embedding_quality(all_features, all_labels)

        # > visualize
        try:
            visualize_embeddings(train_embeddings, train_labels,
                                          f'/data/videomae_outputs/train_embeddings/train_embeddings_epoch{epoch}.png')
        except Exception as e:
            print(f"Could not visualize: {e}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    # > added
    acc5_supported = None

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        #with torch.cuda.amp.autocast():
        output = model(videos)
        loss = criterion(output, target)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # > edited
        if acc5_supported is None:
            acc5_supported = (output.shape[-1] >= 5)
        if acc5_supported:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            (acc1,) = accuracy(output, target, topk=(1,))
            acc5 = acc1  # placeholder to keep logging/file format unchanged

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    # > added
    acc5_supported = None

    # > DEBUG: collect embeddings
    all_embeddings = []
    all_labels = []
    debug_printed = False

    #for batch in metric_logger.log_every(data_loader, 10, header):
    for batch_idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        # compute output
        #with torch.cuda.amp.autocast():
        # > get features and outputs
        features = model.forward_features(videos)  # [B, 768]
        #output = model(videos)
        output = model.head(model.fc_dropout(features))
        loss = criterion(output, target)

        # > DEBUG: save all embeddings and all labels
        all_embeddings.append(features.cpu().numpy())
        all_labels.append(target.cpu().numpy())

        # > print detailed analysis
        if not debug_printed:
            print(f"\n[TEST - Embeddings Analysis]")
            print(f"  Features shape: {features.shape}")
            print(f"  Features stats: mean={features.mean():.4f}, std={features.std():.4f}")
            print(f"  Features range: [{features.min():.4f}, {features.max():.4f}]")

            # > Similarity
            from torch.nn.functional import cosine_similarity
            for i in range(min(4, len(target))):
                for j in range(i+1, min(4, len(target))):
                    sim = cosine_similarity(features[i:i+1], features[j:j+1])
                    same_class = (target[i] == target[j]).item()
                    print(f"  Sim({i},{j}): {sim.item():.4f} | Same class: {same_class}")

            # > variance
            with torch.no_grad():
                batch_std = features.std(dim=0)
                active_dims = (batch_std > 0.01).sum().item()
                print(f"  Active dimensions: {active_dims}/{features.shape[1]} ({active_dims/features.shape[1]*100:.1f}%)")

            debug_printed = True



        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)



        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # > edited
        if acc5_supported is None:
            acc5_supported = (output.shape[-1] >= 5)

        if acc5_supported:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            (acc1,) = accuracy(output, target, topk=(1,))
            acc5 = acc1  # placeholder to keep logging/file format unchanged

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # > analyze all embeddings after testing
    if all_embeddings:
        import numpy as np
        features_all = np.concatenate(all_embeddings)
        labels_all = np.concatenate(all_labels)

        print(f"\n[Final Feature Analysis - All Test Data]")
        print(f"  Total samples: {len(features_all)}")
        print(f"  Feature dimension: {features_all.shape[1]}")
        print(f"  Average feature std across dims: {features_all.std(axis=0).mean():.6f}")

        if features_all.std(axis=0).mean() < 0.01:
            print(" WARNING: Features have very low variance. model may not be learning")

        analyze_embedding_quality(features_all, labels_all)
        visualize_embeddings(all_embeddings, all_labels)

        # > save embeddings
        np.savez('/data/videomae_outputs/test_embeddings/test_embeddings.npz',
                 embeddings=features_all,
                 labels=labels_all)
        print("  Saved embeddings to /data/videomae_outputs/test_embeddings/test_embeddings.npz")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            #data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=float, sep=',') # > edited
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
