from argparse import ArgumentParser

from models.BLO import create_engine
from models.initialize import load_dataset, load_model, load_optimizer, load_functions, load_config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_root', type=str, help='Path to training data root')
    parser.add_argument('--clf_root', type=str, help='Path to classifier training data root')
    parser.add_argument('--val_root', type=str, help='Path to validation data root')

    # Model type
    parser.add_argument('--model', type=str, help='Model type')

    # General hyperparameters
    parser.add_argument('--num_sample', type=int, help='Number of samples')
    parser.add_argument('--aa', type=str, help='AutoAugment policy')
    parser.add_argument('--train_interpolation', type=str, help='Interpolation method for training')
    parser.add_argument('--repre_dim', type=int, help='Representation dimension')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--byol_tau', type=float, help='Momentum coefficient for BYOL')
    parser.add_argument('--max_epochs', type=int, help='Maximum training epochs')
    parser.add_argument('--warm_up_epoch', type=int, help='Warm-up epochs before main training')
    parser.add_argument('--step_scale', type=float, help='Learning rate step scale')
    parser.add_argument('--alpha', type=float, help='Hyperparameter alpha')
    parser.add_argument('--beta', type=float, help='Hyperparameter beta')
    parser.add_argument('--gamma', type=float, help='Hyperparameter gamma')
    parser.add_argument('--delta', type=float, help='Hyperparameter delta')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')

    # Loader & dataset
    parser.add_argument('--clip_len', type=int, help='Length of each video clip')
    parser.add_argument('--frame_sample_rate', type=int, help='Frame sampling rate for training')
    parser.add_argument('--val_clip_len', type=int, help='Length of validation video clips')
    parser.add_argument('--val_frame_sample_rate', type=int, help='Frame sampling rate for validation')
    parser.add_argument('--num_segment', type=int, help='Number of segments per video')
    parser.add_argument('--crop_size', type=int, help='Crop size for video frames')
    parser.add_argument('--short_side_size', type=int, help='Short side size before cropping')
    parser.add_argument('--new_height', type=int, help='Resized frame height')
    parser.add_argument('--new_width', type=int, help='Resized frame width')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, help='Validation batch size')
    parser.add_argument('--shuffle', type=bool, help='Whether to shuffle dataset')
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers')
    parser.add_argument('--pin_memory', type=bool, help='Enable pin_memory for DataLoader')
    parser.add_argument('--drop_last', type=bool, help='Drop last batch if it is smaller than batch_size')

    # Betty engine config
    parser.add_argument('--blotype', type=str, help='Bilevel optimization strategy (e.g., darts)')
    parser.add_argument('--log_step', type=int, help='Logging step interval')
    parser.add_argument('--retain_graph', type=bool, help='Whether to retain computational graph')
    parser.add_argument('--inner_steps', type=int, help='Number of inner optimization steps')
    parser.add_argument('--total_iter', type=int, help='Total number of training iterations')

    # Koopman self-supervised loss
    parser.add_argument('--static_tau', type=float, help='Temperature for static contrastive loss')
    parser.add_argument('--dynamic_tau', type=float, help='Temperature for dynamic contrastive loss')

    # SwAV-specific
    parser.add_argument('--epsilon', type=float, help='Sinkhorn epsilon for SwAV')
    parser.add_argument('--n_iters', type=int, help='Sinkhorn iterations for SwAV')
    parser.add_argument('--temperature', type=float, help='Temperature for softmax in SwAV')

    # MoCo-specific
    parser.add_argument('--K', type=int, help='Queue length in MoCo')
    parser.add_argument('--pinv_rtol', type=float, help='Relative tolerance for pseudo-inverse in Koopman estimation')

    # Backbone architecture
    parser.add_argument('--backbone', type=str, help='Backbone architecture')
    parser.add_argument('--num_frames', type=int, help='Number of input frames for ViT')
    parser.add_argument('--t_patch_size', type=int, help='Temporal patch size for ViT')
    parser.add_argument('--img_size', type=int, help='Image size for ViT input')

    # MLP head settings
    parser.add_argument('--out_dim', type=int, help='Output dimension of MLP head')
    parser.add_argument('--hidden_dim', type=int, help='Hidden layer dimension in MLP')

    # MoCo additional setting
    parser.add_argument('--rep_dim', type=int, help='Feature dimension for MoCo projection head')

    # Optimizer choice
    parser.add_argument('--optimizer_inner', type=str, help='Inner loop optimizer (sgd, adam, adamw, lars)')
    parser.add_argument('--optimizer_outter', type=str, help='Outer loop optimizer (sgd, adam, adamw, lars)')
    args = parser.parse_args()

    train_loader, cls_loader, val_loader = load_dataset(args)
    model = load_model(args).cuda()
    optimizer_inner, optimizer_outter = load_optimizer(args, model)
    inner_func, outter_func = load_functions(args)
    
    engine = create_engine(model, optimizer_outter, optimizer_inner, args, inner_func, outter_func, train_loader)
    engine.run()