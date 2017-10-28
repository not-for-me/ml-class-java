package kr.notforme.ml.ex2;

import static org.nd4j.linalg.ops.transforms.Transforms.log;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class CostCalculator {
    public static double computeCost(INDArray X, INDArray y, INDArray theta) {
        final int trainingSize = y.length();
        INDArray hypothesis = sigmoid(X.mmul(theta));

        return y.neg().mul(log(hypothesis))
                .sub(
                        Nd4j.ones(X.shape()[0], 1).sub(y)
                            .mul(log(Nd4j.ones(X.shape()[0], 1).sub(hypothesis)))
                ).sumNumber().doubleValue() / trainingSize;
    }

    public static double computeRegCost(INDArray X, INDArray y, INDArray theta, double lambda) {
        final int trainingSize = y.length();
        INDArray hypothesis = sigmoid(X.mmul(theta));

        double regVal = pow(theta.get(NDArrayIndex.interval(1, trainingSize), NDArrayIndex.all()), 2)
                                .sumNumber().doubleValue() * lambda / (2 * trainingSize);

        return y.neg().mul(log(hypothesis))
                .sub(
                        Nd4j.ones(X.shape()[0], 1).sub(y)
                            .mul(log(Nd4j.ones(X.shape()[0], 1).sub(hypothesis)))
                ).sumNumber().doubleValue() / trainingSize + regVal;
    }
}
