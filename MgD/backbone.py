# rewriting by Alan Chen

import torch
from torch import nn
import numpy as np

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet, MGdeformableNet3l, MGdeformableNet4l,\
    MGdeformableNet5l, MGdeformableNet3m, MGdeformableNet4m, MGdeformableNet5m, MGdeformableNet3h, MGdeformableNet4h,\
    MGdeformableNet5h, MGdeformableNet2, MGdeformableNet3
from efficientdet.utils import Anchors


class EfficientDetBackbone(nn.Module):
    def __init__(self, opt, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.opt = opt

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        # Multi-granularity deformable convolution layers #
        if self.opt.mgran_deform_conv:
            self.MGdeformableNet3l = MGdeformableNet3l(self)
            self.MGdeformableNet4l = MGdeformableNet4l(self)
            self.MGdeformableNet5l = MGdeformableNet5l(self)

            self.MGdeformableNet3m = MGdeformableNet3m(self)
            self.MGdeformableNet4m = MGdeformableNet4m(self)
            self.MGdeformableNet5m = MGdeformableNet5m(self)

            self.MGdeformableNet3h = MGdeformableNet3h(self)
            self.MGdeformableNet4h = MGdeformableNet4h(self)
            self.MGdeformableNet5h = MGdeformableNet5h(self)

            self.MGdeformableNet2 = MGdeformableNet2(self)
            self.MGdeformableNet3 = MGdeformableNet3(self)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        # print("----------------------------------------------------------------------")
        # print("Before MGdeformaleNets, info of p3, p4, p5:\n", "type of p3:", type(p3), "type of p4:", type(p4), "type of p5:", type(p5))
        # print("\nlength of pXs:", len(p3), len(p4), len(p5))
        # print("\n p3:\n", p3)
        # print("\np3[0].shape:", p3[0].shape)
        # print("\np3.shape:", p3.shape, "p4.shape:", p4.shape, "p5.shape:", p5.shape)

        # Multi-granularity deformable convolution module #
        if self.opt.mgran_deform_conv:
            print("\033[1;35;0m \n>>>>>>> Multi-granularity deformable convolution module is starting <<<<<<< \033[0m")
            # Low granularity module #
            p3l = self.MGdeformableNet3l(p3) #->
            p4l = self.MGdeformableNet4l(p4)
            p5l = self.MGdeformableNet5l(p5)

            # Medium granularity module #
            p3m = self.MGdeformableNet3m(p3)
            p4m = self.MGdeformableNet4m(p4) #->
            p5m = self.MGdeformableNet5m(p5)

            # High granularity module #
            p3h = self.MGdeformableNet3h(p3)
            p4h = self.MGdeformableNet4h(p4)
            p5h = self.MGdeformableNet5h(p5) #->

            # Medium granularity module # Alternate code #
            # p3m = self.MGdeformableNet2(p3)
            # p4m = self.MGdeformableNet2(p4) #->
            # p5m = self.MGdeformableNet3(p5)
            # High granularity module #
            # p3h = self.MGdeformableNet3(p3)
            # p4h = self.MGdeformableNet2(p4)
            # p5h = self.MGdeformableNet3(p5) #->
            # features made by Multi-granularity deformable convolution module #
            # print("After MGdeformaleNets->", "\np3l.shape:", p3l.shape, "p4l.shape:", p4l.shape, "p5l.shape:", p5l.shape)
            # print("After MGdeformaleNets->", "\np3m.shape:", p3m.shape, "p4m.shape:", p4m.shape, "p5m.shape:", p5m.shape)
            # print("After MGdeformaleNets->", "\np3h.shape:", p3h.shape, "p4h.shape:", p4h.shape, "p5h.shape:", p5h.shape)
            features1 = (p3l, p4l, p5l)
            features2 = (p3m, p4m, p5m)
            features3 = (p3h, p4h, p5h)

        features = (p3, p4, p5)
        # print("********************************************************************")
        # print("Before bifpn, info of feature:\n", "type of features:", type(features))
        # print("\nlength of features:", len(features), "\nshape of features[]:", type(features))
        # print("\n features:\n", features)
        # print("\nfeatures[0].shape:",features[0].shape)

        # featuresx = features + features + features
        # print("#####################################################################")
        # print("Before bifpn, info of featurex:\n", "type of featuresx:", type(featuresx))
        # print("\nlength of featuresx:", len(featuresx), "\ntype of features:", type(featuresx))
        # #print("\n featuresx:\n", featuresx)
        # print("\nfeaturesx[0].shape:", featuresx[0].shape)
        features = self.bifpn(features)
        features1 = self.bifpn(features1)
        features2 = self.bifpn(features2)
        features3 = self.bifpn(features3)

        # print("# --------------------------------------------------------------------- #")
        # print("After bifpn, info of feature:\n", "type of features:", type(features))
        # print("\nlength of features:", len(features), "\ntype of features:", type(features))
        # print("\n features:\n", features)
        # print("\nfeatures[0].shape:", features[0].shape)
        # print("\ntype of features[0]:", type(features[0]))
        # print("\nfeatures.shape:", features.shape)

        features = (features[0], features[1], features[2], features[3], features[4],
                    features1[0], features1[1], features1[2], features1[3], features1[4],
                    features2[0], features2[1], features2[2], features2[3], features2[4],
                    features3[0], features3[1], features3[2], features3[3], features3[4])
        """
        # featuresx = self.bifpn(featuresx)
        #
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("After bifpn, info of featurex:\n", "type of featuresx:", type(featuresx))
        # print("\nlength of featuresx:", len(featuresx), "\ntype of featuresx:", type(featuresx))
        # # print("\n features:\n", features)
        # print("\nfeaturesx[0].shape:", featuresx[0].shape)
        """
        regression = self.regressor(features)

        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print("After regressor, info of regression:\n", "type of regression:", type(regression))
        # print("\nLength of regression:", len(regression), "\nType of regression:", type(regression))
        # # print("\n features:\n", features)
        # print("\nRegression.shape:", regression.shape)
        # print("\nRegression[0]:", regression[0])
        # print("\nRegression[-2]:", regression[-2])
        """
        #features = tuple(torch.rand(5,4,112,96,96)) + tuple(torch.rand(5,4,112,96,96)) + features
        #print("list(features).shape:",list(features).shape)

        featurex = features1 + features2 + features3
        regressions = self.regressor(featurex)

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("After regressor, info of regressions:\n", "Type of regressions:", type(regressions))
        print("\nLength of regressions:", len(regressions), "\nType of regressions:", type(regressions))
        # print("\n features:\n", features)
        print("\nRegressions[0].shape:", regressions[0].shape)
        #print("\nregressions.shape:", regressions.shape)
        print("\nRegressions[-2]:", regressions[-2])
        """
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
