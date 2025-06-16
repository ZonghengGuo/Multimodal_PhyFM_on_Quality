import argparse
from preprocessing import processor


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Preprocessing Stage')

    # -------------------------------- Preprocessing Group--------------------------------
    preprocess_args = parser.add_argument_group('Data preprocessing parameters')
    preprocess_args.add_argument('--stage', type=str, choices=['process', 'pair'], help='The stage of preprocessing')
    preprocess_args.add_argument('--dataset_name', type=str, help='name of dataset')
    preprocess_args.add_argument('--raw_data_path', type=str, help='list of dataset input paths')
    preprocess_args.add_argument('--seg_save_path', type=str, help='where to save segmented data')
    preprocess_args.add_argument('--qua_save_path', type=str, help='where to save quality data')
    preprocess_args.add_argument('--pair_save_path', type=str, help='where to save pair data')
    preprocess_args.add_argument('--l_freq', type=float, default=1.0, help='Low-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--h_freq', type=float, default=30.0, help='high-frequency cutoff frequency of bandpass filters (Hz)')
    preprocess_args.add_argument('--rsfreq', type=int, default=300, help='resampling rate (Hz)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print("Start processing dataset")

    if args.dataset_name == 'mimic':
        database = processor.MimicProcessor(args)
        if args.stage == 'process':
            database.process_record()
        elif args.stage == 'pair':
            database.get_data_pair()
        else:
            raise NotImplementedError

    elif args.dataset_name == 'vitaldb':
        database = processor.VitaldbProcessor(args)
        if args.stage == 'process':
            database.process_record()
        elif args.stage == 'pair':
            database.get_data_pair()
        else:
            raise NotImplementedError