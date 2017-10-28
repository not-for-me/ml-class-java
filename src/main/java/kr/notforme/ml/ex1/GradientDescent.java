package kr.notforme.ml.ex1;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GradientDescent {

    public static INDArray runGradientDescent(
            INDArray X, INDArray y, INDArray theta, double alpha, int iterations) {
        System.out.println("Initial Theta: " + theta);

        List<INDArray> costHistories = new ArrayList<>();
        final int trainingSize = y.length();

        for (int i = 0; i < iterations; i++) {
            INDArray hypothesis = X.mmul(theta);
            INDArray error = hypothesis.subi(y);

            // Each Row Version
            double delta0 = alpha * error.mul(X.getColumn(0)).sumNumber().doubleValue() / trainingSize;
            double delta1 = alpha * error.mul(X.getColumn(1)).sumNumber().doubleValue() / trainingSize;

            theta = Nd4j.vstack(theta.getRow(0).sub(delta0), theta.getRow(1).sub(delta1));
            costHistories.add(theta);
        }

        return theta;
    }

    public static INDArray runGradientDescentMulti(
            INDArray X, INDArray y, INDArray theta, double alpha, int iterations) {
        System.out.println("Initial Theta: " + theta);

        List<INDArray> costHistories = new ArrayList<>();
        final int trainingSize = y.length();

        for (int i = 0; i < iterations; i++) {
            INDArray hypothesis = X.mmul(theta);
            INDArray error = hypothesis.subi(y);

            // Vectorization Version
            INDArray delta = X.mulColumnVector(error).sum(0).mul(alpha).div(trainingSize);

            theta.subi(delta.transpose());

            costHistories.add(theta);
        }

        return theta;
    }
}
