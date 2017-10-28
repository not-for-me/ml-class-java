package kr.notforme.ml.ex2;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GradientDescent {

    public static INDArray runGradientDescent
            (INDArray X, INDArray y, INDArray theta, double alpha, int iterations) {

        List<INDArray> costHistories = new ArrayList<>();
        final int trainingSize = y.length();

        for (int i = 0; i < iterations; i++) {
            // ex1과 유일한 차이
            INDArray hypothesis = sigmoid(X.mmul(theta));
            INDArray error = hypothesis.subi(y);

            INDArray delta = X.mulColumnVector(error).sum(0).mul(alpha).div(trainingSize);

            theta.subi(delta.transpose());

            costHistories.add(theta);
        }
        return theta;
    }

    public static INDArray runRegGradientDescent(INDArray X, INDArray y, INDArray theta, int iterations,
                                                 double lambda) {

        List<INDArray> costHistories = new ArrayList<>();
        final int trainingSize = y.length();

        INDArray delta;
        for (int i = 0; i < iterations; i++) {
            // ex1과 유일한 차이
            INDArray hypothesis = sigmoid(X.mmul(theta));

            double delta_0 = hypothesis.sub(y).mul(X.getColumn(0)).sumNumber().doubleValue() / trainingSize;
            delta = Nd4j.create(new double[] { delta_0 }, new int[] { 1, 1 });

            final int featureCnt = theta.shape()[0];
            for (int f = 1; f < featureCnt; f++) {
                double grad_f = hypothesis.sub(y).mul(X.getColumn(f)).sumNumber().doubleValue() / trainingSize
                                + lambda * theta.getRow(f).sumNumber().doubleValue() / trainingSize;

                delta = Nd4j.vstack(delta, Nd4j.create(new double[] { grad_f }, new int[] { 1, 1 }));
            }
            theta.subi(delta);

            costHistories.add(theta);
        }
        return theta;
    }
}
