import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Pretraining Stage')

    # -------------------------------- Preprocessing Group--------------------------------
    preprocess_args = parser.add_argument_group('Pretraining parameters')
    preprocess_args.add_argument('--pair_save_path', type=str, help='where to save pair data')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    pair_path = [
        "/path/to/save1",
        "/path/to/save2"
    ]



