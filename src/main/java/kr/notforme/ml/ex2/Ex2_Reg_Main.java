package kr.notforme.ml.ex2;

import static kr.notforme.ml.ex2.CostCalculator.computeRegCost;
import static kr.notforme.ml.ex2.GradientDescent.runRegGradientDescent;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

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
import org.nd4j.linalg.io.ClassPathResource;

public class Ex2_Reg_Main {
    public static void main(String[] args) throws IOException {
        new Ex2_Reg_Main().runMain();
    }

    private void runMain() throws IOException {
        INDArray data = loadData();
        System.out.println("Data size: " + Arrays.toString(data.shape()));

        INDArray X = data.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray y = data.getColumn(2);
        System.out.println("X size: " + Arrays.toString(X.shape()));
        System.out.println("y size: " + Arrays.toString(y.shape()));

        X = mapFeature(X.getColumn(0), X.getColumn(1));
        System.out.println("X size: " + Arrays.toString(X.shape()));

        final int trainingSize = y.length();
        System.out.println("Training Size: " + trainingSize);

        INDArray theta = Nd4j.zeros(X.shape()[1], 1);
        // Expected 0.693
        System.out.println("Test Cost: " + computeRegCost(X, y, theta, 1));

        INDArray grad = runRegGradientDescent(X, y, theta, 1, 1);
        // Expected first five value 0.0085, 0.0188, 0.0001, 0.0503, 0.0115
        System.out.println("Grad: " + grad.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        INDArray theta2 = Nd4j.ones(X.shape()[1], 1);
        INDArray grad2 = runRegGradientDescent(X, y, theta2, 1, 10);
        // Expected first five value 0.346, 0.1614, 0.1948, 0.2269, 0.0922
        System.out.println("Grad: " + grad2.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));

        INDArray gdTheta = runRegGradientDescent(X, y, theta, 10000, 1);
        INDArray predicted = predict(gdTheta, X);
        System.out.println("Accuracy: " + predicted.eq(y).meanNumber());
    }

    private INDArray mapFeature(INDArray X1, INDArray X2) {
        final int degree = 6;
        INDArray out = Nd4j.ones(X1.shape()[0]).transpose();
        for (int i = 1; i <= degree; i++) {
            for (int j = 0; j <= i; j++) {
                out = Nd4j.hstack(out, pow(X1, i - j).mul(pow(X2, j)));
            }
        }

        return out;
    }

    private INDArray predict(INDArray theta, INDArray X) {
        INDArray result = sigmoid(X.mmul(theta));
        return result.gte(0.5);
    }

    private INDArray loadData() throws IOException {
        String filename = new ClassPathResource("ex2_reg.txt").getFile().getPath();
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
