import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ModelMetrics {
  [key: string]: number;
}

interface ModelData {
  name: string;
  metrics: ModelMetrics;
  confusionMatrix: number[][];
  classificationReport: string;
}

const models: ModelData[] = [
  {
    name: "LogisticRegression + FFT",
    metrics: {
      Accuracy: 0.6907089347706925,
      "Balanced Accuracy": 0.7757780870552434,
      "Precision (macro)": 0.461383909564426,
      "Recall (macro)": 0.7757780870552434,
      "F1 Score (macro)": 0.4962785595522011,
      "F0.5 Score (macro)": 0.467781383746077,
      "F2 Score (macro)": 0.5745968370086171,
      "Matthews Correlation Coefficient": 0.4655005499056539,
      "Cohen’s Kappa": 0.3973867863750976,
      "Hamming Loss": 0.3092910652293075,
      "Zero-One Loss": 0.3092910652293075,
      "Jaccard Score (macro)": 0.3746898863790261,
    },
    confusionMatrix: [
      [12060, 2836, 1350, 1322, 550],
      [102, 373, 27, 27, 27],
      [133, 22, 1120, 133, 40],
      [6, 3, 10, 143, 0],
      [72, 12, 84, 15, 1425],
    ],
    classificationReport: `
               precision    recall  f1-score   support

           0       0.97      0.67      0.79     18118
           1       0.11      0.67      0.20       556
           2       0.43      0.77      0.55      1448
           3       0.09      0.88      0.16       162
           4       0.70      0.89      0.78      1608

    accuracy                           0.69     21892
   macro avg       0.46      0.78      0.50     21892
weighted avg       0.89      0.69      0.75     21892
    `,
  },
  {
    name: "SVC + FFT",
    metrics: {
      Accuracy: 0.9276448017540654,
      "Balanced Accuracy": 0.8777708768906436,
      "Precision (macro)": 0.6890152161669076,
      "Recall (macro)": 0.8777708768906436,
      "F1 Score (macro)": 0.747695052506641,
      "F0.5 Score (macro)": 0.708680320383751,
      "F2 Score (macro)": 0.8083912066797524,
      "Matthews Correlation Coefficient": 0.794875537767853,
      "Cohen’s Kappa": 0.787524596298652,
      "Hamming Loss": 0.07235519824593459,
      "Zero-One Loss": 0.07235519824593462,
      "Jaccard Score (macro)": 0.640878289053162,
    },
    confusionMatrix: [
      [16888, 364, 472, 328, 66],
      [122, 406, 22, 2, 4],
      [62, 4, 1334, 36, 12],
      [13, 4, 8, 137, 0],
      [38, 3, 23, 1, 1543],
    ],
    classificationReport: `
               precision    recall  f1-score   support

           0       0.99      0.93      0.96     18118
           1       0.52      0.73      0.61       556
           2       0.72      0.92      0.81      1448
           3       0.27      0.85      0.41       162
           4       0.95      0.96      0.95      1608

    accuracy                           0.93     21892
   macro avg       0.69      0.88      0.75     21892
weighted avg       0.95      0.93      0.94     21892
    `,
  },
  {
    name: "GradientBoostingClassifier",
    metrics: {
      Accuracy: 0.6768225835921798,
      "Balanced Accuracy": 0.7589360721174725,
      "Precision (macro)": 0.4377877788832672,
      "Recall (macro)": 0.7589360721174725,
      "F1 Score (macro)": 0.47715313871336135,
      "F0.5 Score (macro)": 0.4452581674783208,
      "F2 Score (macro)": 0.5615822897162994,
      "Matthews Correlation Coefficient": 0.4443817021939008,
      "Cohen’s Kappa": 0.37733805418108757,
      "Hamming Loss": 0.3231774164078202,
      "Zero-One Loss": 0.32317741640782016,
      "Jaccard Score (macro)": 0.35052033095076757,
    },
    confusionMatrix: [
      [11806, 2696, 1542, 1266, 808],
      [114, 388, 27, 21, 6],
      [174, 38, 1102, 74, 60],
      [17, 2, 10, 133, 0],
      [78, 25, 109, 8, 1388],
    ],
    classificationReport: `
               precision    recall  f1-score   support

           0       0.97      0.65      0.78     18118
           1       0.12      0.70      0.21       556
           2       0.39      0.76      0.52      1448
           3       0.09      0.82      0.16       162
           4       0.61      0.86      0.72      1608

    accuracy                           0.68     21892
   macro avg       0.44      0.76      0.48     21892
weighted avg       0.88      0.68      0.74     21892
    `,
  },
  {
    name: "XGBoost + classweight",
    metrics: {
      Accuracy: 0.9710396491869175,
      "Balanced Accuracy": 0.8376472784896526,
      "Precision (macro)": 0.9207513622797954,
      "Recall (macro)": 0.8376472784896526,
      "F1 Score (macro)": 0.8728040858738877,
      "F0.5 Score (macro)": 0.8995720179252892,
      "F2 Score (macro)": 0.850550119830036,
      "Matthews Correlation Coefficient": 0.902881952744885,
      "Cohen’s Kappa": 0.9022354015761795,
      "Hamming Loss": 0.028960350813082406,
      "Zero-One Loss": 0.02896035081308246,
      "Jaccard Score (macro)": 0.7874598238938139,
    },
    confusionMatrix: [
      [17961, 29, 102, 7, 19],
      [186, 351, 17, 0, 2],
      [121, 0, 1304, 16, 7],
      [32, 0, 14, 116, 0],
      [48, 1, 33, 0, 1526],
    ],
    classificationReport: `
               precision    recall  f1-score   support

           0       0.98      0.99      0.99     18118
           1       0.92      0.63      0.75       556
           2       0.89      0.90      0.89      1448
           3       0.83      0.72      0.77       162
           4       0.98      0.95      0.97      1608

    accuracy                           0.97     21892
   macro avg       0.92      0.84      0.87     21892
weighted avg       0.97      0.97      0.97     21892
    `,
  },
  {
    name: "GradientBoostingClassifier",
    metrics: {
      Accuracy: 0.7419030651866063,
      "Balanced Accuracy": 0.6855854738105505,
      "Precision (macro)": 0.5040033131242494,
      "Recall (macro)": 0.6855854738105505,
      "F1 Score (macro)": 0.5290492446058422,
      "F0.5 Score (macro)": 0.5095134737726896,
      "F2 Score (macro)": 0.5781842919873397,
      "Matthews Correlation Coefficient": 0.5075646724596735,
      "Cohen’s Kappa": 0.44955053357093244,
      "Hamming Loss": 0.25809693481339363,
      "Zero-One Loss": 0.2580969348133937,
      "Jaccard Score (macro)": 0.42273312637467964,
    },
    confusionMatrix: [
      [13150, 3372, 709, 497, 389],
      [106, 430, 11, 6, 3],
      [192, 40, 1158, 28, 30],
      [95, 24, 8, 35, 0],
      [60, 44, 25, 11, 1468],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.73      0.83     18117
         1.0       0.11      0.77      0.19       556
         2.0       0.61      0.80      0.69      1448
         3.0       0.06      0.22      0.09       162
         4.0       0.78      0.91      0.84      1608

    accuracy                           0.74     21891
   macro avg       0.50      0.69      0.53     21891
weighted avg       0.90      0.74      0.80     21891
    `,
  },
  {
    name: "XGBoost",
    metrics: {
      Accuracy: 0.9020602073911653,
      "Balanced Accuracy": 0.90354613867644,
      "Precision (macro)": 0.7401821373499418,
      "Recall (macro)": 0.90354613867644,
      "F1 Score (macro)": 0.7777480930079296,
      "F0.5 Score (macro)": 0.7515041168890889,
      "F2 Score (macro)": 0.8282139384116947,
      "Matthews Correlation Coefficient": 0.757665172100488,
      "Cohen’s Kappa": 0.7350295233201611,
      "Hamming Loss": 0.09793979260883467,
      "Zero-One Loss": 0.09793979260883467,
      "Jaccard Score (macro)": 0.6928085311531793,
    },
    confusionMatrix: [
      [16194, 1750, 84, 74, 15],
      [72, 475, 4, 3, 2],
      [41, 5, 1381, 16, 5],
      [14, 1, 10, 137, 0],
      [23, 13, 7, 5, 1560],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.99      0.89      0.94     18117
         1.0       0.21      0.85      0.34       556
         2.0       0.93      0.95      0.94      1448
         3.0       0.58      0.85      0.69       162
         4.0       0.99      0.97      0.98      1608

    accuracy                           0.90     21891
   macro avg       0.74      0.90      0.78     21891
weighted avg       0.96      0.90      0.93     21891
    `,
  },
  {
    name: "SVC",
    metrics: {
      Accuracy: 0.939792608834681,
      "Balanced Accuracy": 0.9132559302086172,
      "Precision (macro)": 0.7174485350201399,
      "Recall (macro)": 0.9132559302086172,
      "F1 Score (macro)": 0.7740060066690291,
      "F0.5 Score (macro)": 0.7355582865921905,
      "F2 Score (macro)": 0.8382400836945155,
      "Matthews Correlation Coefficient": 0.8286674286063658,
      "Cohen’s Kappa": 0.8218450583283573,
      "Hamming Loss": 0.060207391165319084,
      "Zero-One Loss": 0.060207391165319035,
      "Jaccard Score (macro)": 0.6820032681534052,
    },
    confusionMatrix: [
      [17048, 616, 135, 280, 38],
      [88, 450, 11, 5, 2],
      [46, 8, 1356, 34, 4],
      [6, 3, 7, 146, 0],
      [22, 6, 6, 1, 1573],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.99      0.94      0.97     18117
         1.0       0.42      0.81      0.55       556
         2.0       0.90      0.94      0.92      1448
         3.0       0.31      0.90      0.46       162
         4.0       0.97      0.98      0.98      1608

    accuracy                           0.94     21891
   macro avg       0.72      0.91      0.77     21891
weighted avg       0.96      0.94      0.95     21891
    `,
  },
  {
    name: "LogisticRegression",
    metrics: {
      Accuracy: 0.673427435932575,
      "Balanced Accuracy": 0.7672022748714898,
      "Precision (macro)": 0.44568770580354944,
      "Recall (macro)": 0.7672022748714898,
      "F1 Score (macro)": 0.48027130265380896,
      "F0.5 Score (macro)": 0.45094171289777707,
      "F2 Score (macro)": 0.5644182336362513,
      "Matthews Correlation Coefficient": 0.44650017110318047,
      "Cohen’s Kappa": 0.37692113012688133,
      "Hamming Loss": 0.32657256406742496,
      "Zero-One Loss": 0.32657256406742496,
      "Jaccard Score (macro)": 0.3607396515859499,
    },
    confusionMatrix: [
      [11697, 2129, 2480, 1339, 472],
      [128, 370, 33, 14, 11],
      [139, 51, 1064, 145, 49],
      [12, 0, 8, 142, 0],
      [48, 6, 76, 9, 1469],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.65      0.78     18117
         1.0       0.14      0.67      0.24       556
         2.0       0.29      0.73      0.42      1448
         3.0       0.09      0.88      0.16       162
         4.0       0.73      0.91      0.81      1608

    accuracy                           0.67     21891
   macro avg       0.45      0.77      0.48     21891
weighted avg       0.88      0.67      0.74     21891
    `,
  },
  {
    name: "SVC (scratch)",
    metrics: {
      Accuracy: 0.8995020784797405,
      "Balanced Accuracy": 0.8928836437029618,
      "Precision (macro)": 0.6470368232489138,
      "Recall (macro)": 0.8928836437029618,
      "F1 Score (macro)": 0.7115791853496387,
      "F0.5 Score (macro)": 0.6672309924514618,
      "F2 Score (macro)": 0.7907106605873352,
      "Matthews Correlation Coefficient": 0.7447602699004171,
      "Cohen’s Kappa": 0.7263785224031143,
      "Hamming Loss": 0.10049792152025946,
      "Zero-One Loss": 0.10049792152025949,
      "Jaccard Score (macro)": 0.6051238302799056,
    },
    confusionMatrix: [
      [16217, 1065, 389, 335, 111],
      [79, 455, 13, 6, 3],
      [70, 13, 1321, 37, 7],
      [6, 3, 12, 141, 0],
      [23, 6, 17, 5, 1557],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.99      0.90      0.94     18117
         1.0       0.30      0.82      0.43       556
         2.0       0.75      0.91      0.83      1448
         3.0       0.27      0.87      0.41       162
         4.0       0.93      0.97      0.95      1608

    accuracy                           0.90     21891
   macro avg       0.65      0.89      0.71     21891
weighted avg       0.95      0.90      0.92     21891
    `,
  },
  {
    name: "LogisticRegression (scratch)",
    metrics: {
      Accuracy: 0.6735187976794116,
      "Balanced Accuracy": 0.7638291776111619,
      "Precision (macro)": 0.44300161026181895,
      "Recall (macro)": 0.7638291776111619,
      "F1 Score (macro)": 0.4788280647412807,
      "F0.5 Score (macro)": 0.448756341230654,
      "F2 Score (macro)": 0.563425387123545,
      "Matthews Correlation Coefficient": 0.44503377095208385,
      "Cohen’s Kappa": 0.3760953569687957,
      "Hamming Loss": 0.32648120232058836,
      "Zero-One Loss": 0.3264812023205884,
      "Jaccard Score (macro)": 0.35894296099880474,
    },
    confusionMatrix: [
      [11704, 2137, 2448, 1315, 513],
      [128, 368, 35, 14, 11],
      [158, 52, 1054, 138, 46],
      [14, 0, 8, 140, 0],
      [42, 9, 66, 13, 1478],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.65      0.78     18117
         1.0       0.14      0.66      0.24       556
         2.0       0.29      0.73      0.42      1448
         3.0       0.09      0.86      0.16       162
         4.0       0.72      0.92      0.81      1608

    accuracy                           0.67     21891
   macro avg       0.44      0.76      0.48     21891
weighted avg       0.88      0.67      0.74     21891
    `,
  },
  {
    name: "LogisticRegression (scratch) + class_weight",
    metrics: {
      Accuracy: 0.6762596500845096,
      "Balanced Accuracy": 0.763617402596117,
      "Precision (macro)": 0.4447988987649632,
      "Recall (macro)": 0.763617402596117,
      "F1 Score (macro)": 0.4801502465289735,
      "F0.5 Score (macro)": 0.4505431152616162,
      "F2 Score (macro)": 0.5632330624864108,
      "Matthews Correlation Coefficient": 0.4472177467293554,
      "Cohen’s Kappa": 0.3788762697234933,
      "Hamming Loss": 0.3237403499154904,
      "Zero-One Loss": 0.3237403499154904,
      "Jaccard Score (macro)": 0.3605015378745349,
    },
    confusionMatrix: [
      [11770, 2116, 2349, 1380, 502],
      [129, 367, 35, 15, 10],
      [151, 47, 1057, 146, 47],
      [13, 0, 9, 140, 0],
      [47, 7, 69, 15, 1470],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.65      0.78     18117
         1.0       0.14      0.66      0.24       556
         2.0       0.30      0.73      0.43      1448
         3.0       0.08      0.86      0.15       162
         4.0       0.72      0.91      0.81      1608

    accuracy                           0.68     21891
   macro avg       0.44      0.76      0.48     21891
weighted avg       0.88      0.68      0.74     21891
    `,
  },
  {
    name: "SVC + class_weight",
    metrics: {
      Accuracy: 0.9109679777077337,
      "Balanced Accuracy": 0.9111937657928258,
      "Precision (macro)": 0.6677999919104408,
      "Recall (macro)": 0.9111937657928258,
      "F1 Score (macro)": 0.720587815947782,
      "F0.5 Score (macro)": 0.6837331216062761,
      "F2 Score (macro)": 0.7936026922586465,
      "Matthews Correlation Coefficient": 0.7695236411749851,
      "Cohen’s Kappa": 0.7535864967839402,
      "Hamming Loss": 0.08903202229226623,
      "Zero-One Loss": 0.08903202229226626,
      "Jaccard Score (macro)": 0.6286204783405258,
    },
    confusionMatrix: [
      [16427, 833, 201, 594, 62],
      [85, 453, 11, 4, 3],
      [39, 10, 1340, 55, 4],
      [3, 1, 7, 151, 0],
      [22, 6, 9, 0, 1571],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.99      0.91      0.95     18117
         1.0       0.35      0.81      0.49       556
         2.0       0.85      0.93      0.89      1448
         3.0       0.19      0.93      0.31       162
         4.0       0.96      0.98      0.97      1608

    accuracy                           0.91     21891
   macro avg       0.67      0.91      0.72     21891
weighted avg       0.96      0.91      0.93     21891
    `,
  },
  {
    name: "SVC (scratch) + class_weight",
    metrics: {
      Accuracy: 0.9238499840116943,
      "Balanced Accuracy": 0.6689819451241102,
      "Precision (macro)": 0.7587226244681534,
      "Recall (macro)": 0.6689819451241102,
      "F1 Score (macro)": 0.7058180545184115,
      "F0.5 Score (macro)": 0.7348323359687667,
      "F2 Score (macro)": 0.6823808016383841,
      "Matthews Correlation Coefficient": 0.7337891154200804,
      "Cohen’s Kappa": 0.7296925635705074,
      "Hamming Loss": 0.0761500159883057,
      "Zero-One Loss": 0.07615001598830573,
      "Jaccard Score (macro)": 0.5862422630992274,
    },
    confusionMatrix: [
      [17661, 139, 231, 36, 50],
      [374, 174, 7, 0, 1],
      [509, 8, 884, 24, 23],
      [47, 0, 23, 92, 0],
      [172, 2, 21, 0, 1413],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.94      0.97      0.96     18117
         1.0       0.54      0.31      0.40       556
         2.0       0.76      0.61      0.68      1448
         3.0       0.61      0.57      0.59       162
         4.0       0.95      0.88      0.91      1608

    accuracy                           0.92     21891
   macro avg       0.76      0.67      0.71     21891
weighted avg       0.92      0.92      0.92     21891
    `,
  },
  {
    name: "LogisticRegression + class_weight",
    metrics: {
      Accuracy: 0.6722397332236992,
      "Balanced Accuracy": 0.7652304307965235,
      "Precision (macro)": 0.4499269086262361,
      "Recall (macro)": 0.7652304307965235,
      "F1 Score (macro)": 0.4810876346309052,
      "F0.5 Score (macro)": 0.45407228955305506,
      "F2 Score (macro)": 0.5615530547580689,
      "Matthews Correlation Coefficient": 0.44576218873778845,
      "Cohen’s Kappa": 0.3759316350873916,
      "Hamming Loss": 0.32776026677630077,
      "Zero-One Loss": 0.32776026677630077,
      "Jaccard Score (macro)": 0.3633743246232913,
    },
    confusionMatrix: [
      [11680, 2288, 2307, 1418, 424],
      [132, 367, 36, 16, 5],
      [127, 60, 1067, 149, 45],
      [11, 0, 9, 142, 0],
      [59, 5, 77, 7, 1460],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.64      0.78     18117
         1.0       0.13      0.66      0.22       556
         2.0       0.31      0.74      0.43      1448
         3.0       0.08      0.88      0.15       162
         4.0       0.75      0.91      0.82      1608

    accuracy                           0.67     21891
   macro avg       0.45      0.77      0.48     21891
weighted avg       0.88      0.67      0.74     21891
    `,
  },
  {
    name: "XGBoost + classweight",
    metrics: {
      Accuracy: 0.9840573751770134,
      "Balanced Accuracy": 0.8984661259335571,
      "Precision (macro)": 0.9447861927583409,
      "Recall (macro)": 0.8984661259335571,
      "F1 Score (macro)": 0.9202302879830269,
      "F0.5 Score (macro)": 0.9345897355151983,
      "F2 Score (macro)": 0.9068725192151875,
      "Matthews Correlation Coefficient": 0.9469728196540399,
      "Cohen’s Kappa": 0.9467111920761238,
      "Hamming Loss": 0.015942624822986615,
      "Zero-One Loss": 0.015942624822986584,
      "Jaccard Score (macro)": 0.8601187279579083,
    },
    confusionMatrix: [
      [18043, 41, 23, 3, 7],
      [120, 429, 5, 0, 2],
      [57, 2, 1368, 15, 6],
      [20, 0, 12, 130, 0],
      [28, 2, 5, 1, 1572],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.99      1.00      0.99     18117
         1.0       0.91      0.77      0.83       556
         2.0       0.97      0.94      0.96      1448
         3.0       0.87      0.80      0.84       162
         4.0       0.99      0.98      0.98      1608

    accuracy                           0.98     21891
   macro avg       0.94      0.90      0.92     21891
weighted avg       0.98      0.98      0.98     21891
    `,
  },
  {
    name: "GradientBoostingClassifier + classweight",
    metrics: {
      Accuracy: 0.8349093234662647,
      "Balanced Accuracy": 0.8218918160952701,
      "Precision (macro)": 0.5490063111437393,
      "Recall (macro)": 0.8218918160952701,
      "F1 Score (macro)": 0.6085374500631859,
      "F0.5 Score (macro)": 0.5670639868291364,
      "F2 Score (macro)": 0.6894167957696108,
      "Matthews Correlation Coefficient": 0.6179450229626697,
      "Cohen’s Kappa": 0.5881698380294111,
      "Hamming Loss": 0.16509067653373533,
      "Zero-One Loss": 0.16509067653373533,
      "Jaccard Score (macro)": 0.48710124249840947,
    },
    confusionMatrix: [
      [15093, 1220, 708, 694, 402],
      [127, 407, 9, 11, 2],
      [165, 41, 1151, 61, 30],
      [19, 3, 7, 133, 0],
      [61, 32, 18, 4, 1493],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.98      0.83      0.90     18117
         1.0       0.24      0.73      0.36       556
         2.0       0.61      0.79      0.69      1448
         3.0       0.15      0.82      0.25       162
         4.0       0.77      0.93      0.84      1608

    accuracy                           0.83     21891
   macro avg       0.55      0.82      0.61     21891
weighted avg       0.91      0.83      0.86     21891
    `,
  },
  {
    name: "RandomForest (scratch)",
    metrics: {
      Accuracy: 0.37307569320725414,
      "Balanced Accuracy": 0.6642294072190486,
      "Precision (macro)": 0.396557300405158,
      "Recall (macro)": 0.6642294072190486,
      "F1 Score (macro)": 0.35077174695313473,
      "F0.5 Score (macro)": 0.3544425863490387,
      "F2 Score (macro)": 0.4135490966167267,
      "Matthews Correlation Coefficient": 0.27579109013055675,
      "Cohen’s Kappa": 0.17247935359536481,
      "Hamming Loss": 0.6269243067927459,
      "Zero-One Loss": 0.6269243067927459,
      "Jaccard Score (macro)": 0.24275319928214273,
    },
    confusionMatrix: [
      [5401, 7536, 3385, 1347, 448],
      [65, 426, 47, 15, 3],
      [86, 237, 841, 140, 144],
      [6, 16, 6, 134, 0],
      [61, 79, 97, 6, 1365],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.96      0.30      0.46     18117
         1.0       0.05      0.77      0.10       556
         2.0       0.19      0.58      0.29      1448
         3.0       0.08      0.83      0.15       162
         4.0       0.70      0.85      0.77      1608

    accuracy                           0.37     21891
   macro avg       0.40      0.66      0.35     21891
weighted avg       0.86      0.37      0.46     21891
    `,
  },
  {
    name: "RandomForestClassifier",
    metrics: {
      Accuracy: 0.5743456214882828,
      "Balanced Accuracy": 0.6173711285281352,
      "Precision (macro)": 0.3595278305380497,
      "Recall (macro)": 0.6173711285281352,
      "F1 Score (macro)": 0.37367623924606236,
      "F0.5 Score (macro)": 0.35741251850421407,
      "F2 Score (macro)": 0.4369354448030248,
      "Matthews Correlation Coefficient": 0.2871629505350346,
      "Cohen’s Kappa": 0.23249071742557337,
      "Hamming Loss": 0.42565437851171717,
      "Zero-One Loss": 0.42565437851171717,
      "Jaccard Score (macro)": 0.259730366900491,
    },
    confusionMatrix: [
      [10363, 4215, 1344, 1038, 1157],
      [143, 358, 33, 11, 11],
      [566, 249, 369, 81, 183],
      [31, 4, 2, 125, 0],
      [131, 26, 91, 2, 1358],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.92      0.57      0.71     18117
         1.0       0.07      0.64      0.13       556
         2.0       0.20      0.25      0.22      1448
         3.0       0.10      0.77      0.18       162
         4.0       0.50      0.84      0.63      1608

    accuracy                           0.57     21891
   macro avg       0.36      0.62      0.37     21891
weighted avg       0.82      0.57      0.65     21891
    `,
  },
  {
    name: "GradientBoostingClassifier (scratch)",
    metrics: {
      Accuracy: 0.7318989539079988,
      "Balanced Accuracy": 0.7854447109061453,
      "Precision (macro)": 0.5001860564593004,
      "Recall (macro)": 0.7854447109061453,
      "F1 Score (macro)": 0.5296864561625189,
      "F0.5 Score (macro)": 0.5063754505753285,
      "F2 Score (macro)": 0.5941226483578085,
      "Matthews Correlation Coefficient": 0.4940952649066944,
      "Cohen’s Kappa": 0.4377904650514901,
      "Hamming Loss": 0.2681010460920013,
      "Zero-One Loss": 0.26810104609200125,
      "Jaccard Score (macro)": 0.4125467810607185,
    },
    confusionMatrix: [
      [13005, 2146, 816, 1791, 359],
      [117, 397, 14, 19, 9],
      [160, 19, 1051, 143, 75],
      [9, 2, 8, 143, 0],
      [129, 13, 33, 7, 1426],
    ],
    classificationReport: `
               precision    recall  f1-score   support

         0.0       0.97      0.72      0.82     18117
         1.0       0.15      0.71      0.25       556
         2.0       0.55      0.73      0.62      1448
         3.0       0.07      0.88      0.13       162
         4.0       0.76      0.89      0.82      1608

    accuracy                           0.73     21891
   macro avg       0.50      0.79      0.53     21891
weighted avg       0.90      0.73      0.79     21891
    `,
  },
];

const keyMetrics = [
  "Accuracy",
  "Balanced Accuracy",
  "F1 Score (macro)",
  "Matthews Correlation Coefficient",
];

const BenchmarkAllTable: React.FC = () => {
  const [sortMetric, setSortMetric] = useState<string>("Accuracy");
  const [selectedModel, setSelectedModel] = useState<ModelData | null>(null);

  const sortedModels = useMemo(() => {
    return [...models].sort((a, b) => b.metrics[sortMetric] - a.metrics[sortMetric]);
  }, [sortMetric]);

  const openModal = (model: ModelData) => setSelectedModel(model);
  const closeModal = () => setSelectedModel(null);

  return (
    <div className="p-6 max-w-7xl mx-auto bg-black text-white">
      <h1 className="text-3xl font-bold mb-6 text-yellow-500">
        Model Benchmark Results
      </h1>

      <div className="mb-6 flex items-center">
        <label className="mr-3 text-lg">Sort by:</label>
        <Select value={sortMetric} onValueChange={setSortMetric}>
          <SelectTrigger className="w-[200px] bg-gray-800 text-white border-gray-600 focus:ring-yellow-500">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="bg-gray-800 text-white border-gray-600">
            {keyMetrics.map((metric) => (
              <SelectItem key={metric} value={metric}>
                {metric}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sortedModels.map((model) => (
          <motion.div
            key={model.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            whileHover={{ scale: 1.03 }}
            className="bg-gray-900 rounded-lg shadow-lg overflow-hidden"
          >
            <div className="p-4">
              <h2 className="text-xl font-semibold text-yellow-500 mb-4">
                {model.name}
              </h2>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {keyMetrics.map((metric) => (
                  <div key={metric} className="flex justify-between">
                    <span>{metric}:</span>
                    <span>{model.metrics[metric].toFixed(4)}</span>
                  </div>
                ))}
              </div>
              <Button
                onClick={() => openModal(model)}
                className="mt-4 w-full bg-yellow-600 text-white hover:bg-yellow-500"
              >
                View Details
              </Button>
            </div>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {selectedModel && (
          <Dialog open={!!selectedModel} onOpenChange={closeModal}>
            <DialogContent className="bg-gray-800 text-white max-w-2xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle className="text-2xl text-yellow-500">
                  {selectedModel.name}
                </DialogTitle>
              </DialogHeader>

              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Metrics</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries(selectedModel.metrics).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span>{key}:</span>
                        <span>{value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="text-white">True \ Pred</TableHead>
                        {["0", "1", "2", "3", "4"].map((cls) => (
                          <TableHead key={cls} className="text-white text-center">
                            {cls}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedModel.confusionMatrix.map((row, i) => (
                        <TableRow key={i}>
                          <TableCell className="text-white">{i}</TableCell>
                          {row.map((val, j) => (
                            <TableCell key={j} className="text-white text-center">
                              {val}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">Classification Report</h3>
                  <pre className="text-sm bg-gray-700 p-4 rounded whitespace-pre-wrap">
                    {selectedModel.classificationReport.trim()}
                  </pre>
                </div>
              </div>

              <DialogFooter>
                <Button
                  onClick={closeModal}
                  className="w-full bg-yellow-500 text-white hover:bg-yellow-400"
                >
                  Close
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </AnimatePresence>
    </div>
  );
};

export default BenchmarkAllTable;