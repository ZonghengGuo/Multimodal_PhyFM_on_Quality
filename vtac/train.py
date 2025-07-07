from torch.utils.data import DataLoader
from vtac.nets import *
from vtac.tools import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import sklearn
from models.Transformer import MultiModalTransformerQuality
from models.ResNet import MultiModalResNetQuality
from models.Mamba import MultiModalMambaQuality
from models.PWSA import MultiModalLongformerQuality


class VtacTrainer:
    def __init__(self, args):
        self.train_path = os.path.join(args.dataset_name, "out/lead_selected/train.pt")
        self.val_path = os.path.join(args.dataset_name, "out/lead_selected/val.pt")
        self.test_path = os.path.join(args.dataset_name, "out/lead_selected/test.pt")
        self.backbone = args.backbone
        self.out_dim = args.out_dim
        self.window_size = args.min_lr
        self.batch_size = args.batch_size

    def training(self):
        SEED = 1
        os.environ["PYTHONHASHSEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load preprocessed dataset
        trainset_x, trainset_y = torch.load(self.train_path, weights_only=True)
        valset_x, valset_y = torch.load(self.val_path, weights_only=True)
        testset_x, testset_y = torch.load(self.test_path, weights_only=True)

        zero_nans = lambda x: torch.nan_to_num(x, 0)

        trainset_x = zero_nans(trainset_x)
        testset_x = zero_nans(testset_x)
        valset_x = zero_nans(valset_x)

        batch_size = self.batch_size
        lr = 0.0001
        dropout_probability = 0.1
        positive_class_weight = 4

        if self.backbone == "pwsa":
            backbone = MultiModalLongformerQuality(2, self.out_dim, 4, 2, 256, self.window_size)
        elif self.backbone == 'transformer':
            backbone = MultiModalTransformerQuality(2, self.out_dim, 4, 2, 256)
        elif self.backbone == 'resnet':
            backbone = MultiModalResNetQuality(2, self.out_dim, 18)
        elif self.backbone == 'mamba':
            backbone = MultiModalMambaQuality(2, self.out_dim, 2, 256)
        else:
            raise ValueError(
                f"Unsupported backbone: '{self.backbone}'. Please choose from ['pwas', 'resnet', 'transformer', 'mamba'].")

        params_training = {
            "framework": "self.backbone",
            "weighted_class": positive_class_weight,
            "learning_rate": lr,
            "adam_weight_decay": 0.005,
            "batch_size": batch_size,
            "max_epoch": 500,
            "data_length": 2500,
        }


        # save path of trained model
        tuning_name = (
            f"{batch_size}-{lr}-{dropout_probability}-{positive_class_weight}-{SEED}"
        )

        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", tuning_name
        )

        if not any(
                os.path.exists(os.path.join(model_path, x)) for x in ["", "auc", "score"]
        ):
            # os.makedirs(model_path)
            os.makedirs(os.path.join(model_path, "auc"))
            os.makedirs(os.path.join(model_path, "score"))
        save_path = os.path.join(model_path, "results.txt")
        logger = get_logger(logpath=save_path, filepath=os.path.abspath(__file__))
        logger.info(params_training)

        model_save_path = os.path.join(
            model_path, str(params_training["learning_rate"]) + ".pt"
        )

        dataset_train = Dataset_train(trainset_x, trainset_y)
        dataset_eval = Dataset_train(valset_x, valset_y)
        dataset_test = Dataset_train(testset_x, testset_y)

        params = {
            "batch_size": params_training["batch_size"],
            "shuffle": False,
            "num_workers": 0,
        }

        iterator_train = DataLoader(dataset_train, **params)
        iterator_test = DataLoader(dataset_eval, **params)

        # Todo: change it into real path
        checkpoint = torch.load(f"model_saved/{self.backbone}_teacher.pth")
        backbone.load_state_dict(checkpoint["model_state_dict"])
        encoder = backbone.encoder

        print(f"Load model {self.backbone} successfully!!!")

        for param in encoder.parameters():
            param.requires_grad = True

        model = FinetuneModel(pre_trained_encoder=encoder, num_classes=1)

        logger.info(model)
        logger.info(
            "Num of Parameters: {}M".format(
                sum(x.numel() for x in model.parameters()) / 1000000
            )
        )

        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params_training["learning_rate"],
            weight_decay=params_training["adam_weight_decay"],
        )  # optimize all cnn parameters
        loss_ce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([params_training["weighted_class"]]).to(device)
        )

        num_epochs = params_training["max_epoch"]

        results_trainloss = []
        results_evalloss = []
        results_score = []
        results_TPR = []
        results_TNR = []
        results_acc = []
        max_score, max_auc = 0, 0
        min_eval_loss = float("inf")

        for t in range(1, 1 + num_epochs):
            train_loss = 0
            model = model.train()
            train_TP, train_FP, train_TN, train_FN = 0, 0, 0, 0

            for b, batch in enumerate(
                    iterator_train, start=1
            ):  # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
                loss, Y_train_prediction, y_train = train_model(
                    batch,
                    model,
                    loss_ce,
                    device,
                    weight=0
                )

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss /= b
            eval_loss = 0
            model = model.eval()
            types_TP = 0
            types_FP = 0
            types_TN = 0
            types_FN = 0
            with torch.no_grad():
                for b, batch in enumerate(iterator_test, start=1):
                    loss, Y_eval_prediction, y_test = eval_model(
                        batch, model, loss_ce, device
                    )
                    types_TP, types_FP, types_TN, types_FN = evaluation_test(
                        Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN
                    )
                    eval_loss += loss.item()

            eval_loss /= b
            acc = 100 * (types_TP + types_TN) / (types_TP + types_TN + types_FP + types_FN)
            score = (
                    100
                    * (types_TP + types_TN)
                    / (types_TP + types_TN + types_FP + 5 * types_FN)
            )
            TPR = 100 * types_TP / (types_TP + types_FN)
            TNR = 100 * types_TN / (types_TN + types_FP)

            if types_TP + types_FP == 0:
                ppv = 1
            else:
                ppv = types_TP / (types_TP + types_FP)

            auc = sklearn.metrics.roc_auc_score(
                y_test.cpu().detach().numpy(), Y_eval_prediction.cpu().detach().numpy()
            )
            f1 = types_TP / (types_TP + 0.5 * (types_FP + types_FN))
            sen = types_TP / (types_TP + types_FN)
            spec = types_TN / (types_TN + types_FP)

            if auc > max_auc:
                max_auc = auc
                torch.save(
                    model.state_dict(), os.path.join(model_path, "auc", str(t) + ".pt")
                )

            if score > max_score:
                max_score = score
                torch.save(
                    model.state_dict(), os.path.join(model_path, "score", str(t) + ".pt")
                )

            logger.info(20 * "-")

            logger.info(params_training["framework"] + " Epoch " + str(t))

            logger.info(
                "total_loss: "
                + str(round(train_loss, 5))
                + " train_loss: "
                + str(round(train_loss, 5))
                + " eval_loss: "
                + str(round(eval_loss, 5))
            )

            logger.info(
                "TPR: "
                + str(round(TPR, 3))
                + " TNR: "
                + str(round(TNR, 3))
                + " Score: "
                + str(round(score, 3))
                + " Acc: "
                + str(round(acc, 3))
            )

            logger.info(
                "PPV: "
                + str(round(ppv, 3))
                + " AUC: "
                + str(round(auc, 3))
                + " F1: "
                + str(round(f1, 3))
            )
