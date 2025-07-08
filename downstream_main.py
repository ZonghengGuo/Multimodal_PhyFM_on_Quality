import argparse
from vtac.preprocess import preprocess_vtac
from vtac.train import VtacTrainer
from AF.preprocess import AFProcessor
from AF.train import AfTrainer


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Downstream Stage.')

    # -------------------------------- Downstream Group--------------------------------
    args = parser.add_argument_group('Downstream Tasks.')
    args.add_argument('--dataset_name', type=str, help='dataset name')
    args.add_argument('--stage', type=str, help='stage name')
    args.add_argument('--raw_data_path', type=str, help='vtac dataset input paths')
    parser.add_argument('--backbone', type=str, help='The architecture of the feature extractor')
    # vtac_args.add_argument('--save_path', type=str, help='vtac dataset save path')
    args.add_argument('--sampling_rate', type=int, default=300, help='sampling rate for vtac')
    args.add_argument('--powerline_frequency', type=int, default=60, help='sampling rate for vtac')
    parser.add_argument('--out_dim', type=int, default=512, help='Output feature dimension.')
    parser.add_argument('--min_lr', type=int, default=8,
                        help='The window size of physiological windowed sparse attention.')
    parser.add_argument('--model_save_path', type=str, default="model_saved",
                        help='Path to the directory where trained models will be saved.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of training.')
    parser.add_argument('--rsfreq', type=int, default=300, help='resampling rate (Hz)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.dataset_name == "vtac":
        if args.stage == "preprocessing":
            processor = preprocess_vtac(args)
            processor.preprocess_save()
            processor.splitting()
        elif args.stage == "training":
            trainer = VtacTrainer(args)
            trainer.training()

    elif args.dataset_name == "af":
        if args.stage == "preprocessing":
            processor = AFProcessor(args)
            processor.process_save()
        elif args.stage == "training":
            trainer = AfTrainer(args)
            trainer.training()











