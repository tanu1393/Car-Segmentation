import torch
from torch import nn
from torchmetrics.classification import Dice
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
import segmentation_models_pytorch as smp
import argparse

from dataset import SegmentDataset


def get_train_val_data(dirPath=r'/content/car-segmentation', batch_size=4, val_data_proportion=0.2):
    car_data = SegmentDataset(dirPath = dirPath)

    train_inds = range(int(len(car_data)*(1-val_data_proportion)))
    val_inds = range(int(len(car_data)*(1-val_data_proportion)), len(car_data))

    trainData = Subset(car_data, train_inds)
    valData = Subset(car_data, val_inds)

    train_dl = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(valData, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_dl, val_dl


class CarPartSegmentation(pl.LightningModule):

    def __init__(self, encoder_name, in_channels, out_classes, T_MAX, **kwargs):
        super().__init__()
        self.T_MAX = T_MAX
        self.out_classes = out_classes
        self.model = smp.Unet(
            encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        
        self.dice_metric = Dice(num_classes=out_classes, threshold=0.7)
        self.criterion = nn.CrossEntropyLoss()
        
        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask, _, _, _ = batch
        
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        
        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        logits_mask = self.forward(image)
        
        loss_ce = self.criterion(logits_mask, mask.long())
        
        prob_mask = logits_mask.argmax(axis=1)

        dice_metric = self.dice_metric(prob_mask, mask)
        
        self.log(f"{stage}_loss", loss_ce, prog_bar=True)
        self.log(f"{stage}_dice", dice_metric, prog_bar=True)

        return {
            "loss": loss_ce,
            "dice_metric": dice_metric,
        }

    def shared_epoch_end(self, outputs, stage):
        pass

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        # empty set output list
        self.training_step_outputs.clear()
        return 

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()
        return 

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        # empty set output list
        self.test_step_outputs.clear()
        return 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, amsgrad=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
        return
    
    def configure_callbacks(self):
        early_stop = pl.callbacks.EarlyStopping(monitor="valid_loss", mode="min", min_delta=0.001, patience=5)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid_loss", filename='{epoch}-{valid_loss:.3f}')
        return [early_stop, checkpoint]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Segmentation training')
    parser.add_argument('--dirPath', default='/car-segmentation', help='Parent directory of training dataset')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--pretrained_model_name', default='resnet34', help='Pretrained encoder model name')
    parser.add_argument('--model_save_path', default='/trained_model', help='Path to save trained model')
    args = parser.parse_args()

    # Some training hyperparameters
    EPOCHS = args.epochs
    OUT_CLASSES = 5

    train_dl, val_dl = get_train_val_data(dirPath=args.dirPath)
    T_MAX = EPOCHS * len(train_dl)

    model = CarPartSegmentation(args.pretrained_model_name, in_channels=3, out_classes=5, activation='sigmoid', T_MAX=T_MAX)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl,
    )

    del model

    best_model = CarPartSegmentation.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        encoder_name=args.pretrained_model_name, 
        in_channels=3, 
        out_classes=OUT_CLASSES, 
        activation='sigmoid',
        T_MAX=T_MAX
        )
    
    best_model.model.save_pretrained(args.model_save_path)
    print(f"Best model saved at {args.model_save_path}")