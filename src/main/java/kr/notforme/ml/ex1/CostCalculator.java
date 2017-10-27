package kr.notforme.ml.ex1;

import org.nd4j.linalg.api.ndarray.INDArray;

public class CostCalculator {
    public static double computeCost(INDArray X, INDArray y, INDArray theta) {
        final int trainingSize = y.length();
        INDArray hypothesis = X.mmul(theta);

        INDArray error = hypothesis.sub(y);
        INDArray squaredError = error.mul(error);

        return squaredError.sumNumber().doubleValue() / (2 * trainingSize);
    }
}
