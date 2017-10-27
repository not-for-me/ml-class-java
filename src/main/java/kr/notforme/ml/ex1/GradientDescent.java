package kr.notforme.ml.ex1;

import static kr.notforme.ml.support.PrintUtil.printDim;

import org.nd4j.linalg.api.ndarray.INDArray;

public class GradientDescent {

    public static INDArray runGradientDescent(
            INDArray X, INDArray y, INDArray theta, double alpha, int iterations) {
        System.out.println("Initial Theta: " + theta);

        final int trainingSize = y.length();

        for (int i = 0; i < iterations; i++) {
            INDArray hypothesis = X.mmul(theta);
            INDArray error = hypothesis.subi(y);

            // Each Row Version
//            double delta0 = alpha * error.mul(X.getColumn(0)).sumNumber().doubleValue() / trainingSize;
//            double delta1 = alpha * error.mul(X.getColumn(1)).sumNumber().doubleValue() / trainingSize;
            // Vectorize Version
            INDArray delta = X.mulColumnVector(error).sum(0).div(trainingSize).mul(alpha);

            theta.subi(delta.transpose());
        }

        return theta;
    }

}
