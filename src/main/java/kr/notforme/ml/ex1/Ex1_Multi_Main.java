package kr.notforme.ml.ex1;

import static kr.notforme.ml.ex1.GradientDescent.runGradientDescentMulti;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.io.ClassPathResource;

import kr.notforme.ml.ex1.FeatureNormalizer.Norm;

public class Ex1_Multi_Main {
    public static void main(String[] args) throws IOException {
        new Ex1_Multi_Main().runMain();
    }

    private void runMain() throws IOException {
        INDArray data = loadEx1_Multi();
        System.out.println("Data size: " + Arrays.toString(data.shape()));

        INDArray X = data.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray y = data.getColumn(2);
        System.out.println("X size: " + Arrays.toString(X.shape()));
        System.out.println("y size: " + Arrays.toString(y.shape()));

        final int trainingSize = y.length();
        System.out.println("Training Size: " + trainingSize);

        Norm norm = FeatureNormalizer.norm(X);
        X = norm.getResult();

        INDArray ones = Nd4j.ones(trainingSize).transpose();
        X = Nd4j.hstack(ones, X);
        System.out.println("X size: " + Arrays.toString(X.shape()));

        INDArray theta = Nd4j.zeros(3, 1);
        final double alpha = 0.01;
        final int iterations = 1000;
        theta = runGradientDescentMulti(X, y, theta, alpha, iterations);

        System.out.println("Trained Theta: " + theta);

        System.out.println("Prediction 1650ft^2, 3br");
        INDArray input = norm.getNormValue(Nd4j.create(new float[] { 1650, 3 }, new int[] { 1, 2 }));
        input = Nd4j.hstack(Nd4j.ones(1), input);

        INDArray predicted = input.mmul(theta);
        System.out.println("Predicted: " + predicted.sumNumber());

        // 직접 미분으로 theta 계산
        INDArray theta_eqn = InvertMatrix.invert(X.transpose().mmul(X), false).mmul(X.transpose()).mmul(y);
        System.out.println("Eqn Theta: " + theta_eqn);

        System.out.println("Eqn Theta Prediction 1650ft^2, 3br");
        INDArray predicted_eqn = input.mmul(theta_eqn);
        System.out.println("Predicted Eqn: " + predicted_eqn.sumNumber());

    }

    private INDArray loadEx1_Multi() throws IOException {
        String filename = new ClassPathResource("ex1_multi.txt").getFile().getPath();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = "";
        List<Double> vals = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            // use comma as separator
            String[] splited = line.split(",");
            vals.add(Double.valueOf(splited[0]));
            vals.add(Double.valueOf(splited[1]));
            vals.add(Double.valueOf(splited[2]));
        }

        Double[] valArr = vals.toArray(new Double[vals.size()]);
        return Nd4j.create(ArrayUtils.toPrimitive(valArr), new int[] { vals.size() / 3, 3 });
    }
}
