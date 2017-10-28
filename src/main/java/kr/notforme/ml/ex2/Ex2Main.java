package kr.notforme.ml.ex2;

import static kr.notforme.ml.ex2.CostCalculator.computeCost;
import static kr.notforme.ml.ex2.GradientDescent.runGradientDescent;
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

public class Ex2Main {
    public static void main(String[] args) throws IOException {
        new Ex2Main().runMain();
    }

    private void runMain() throws IOException {
        INDArray data = loadData();
        System.out.println("Data size: " + Arrays.toString(data.shape()));

        INDArray X = data.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray y = data.getColumn(2);
        System.out.println("X size: " + Arrays.toString(X.shape()));
        System.out.println("y size: " + Arrays.toString(y.shape()));

        final int trainingSize = y.length();
        System.out.println("Training Size: " + trainingSize);

        INDArray ones = Nd4j.ones(trainingSize).transpose();
        System.out.println("Zeros size: " + Arrays.toString(ones.shape()));
        X = Nd4j.hstack(ones, X);
        System.out.println("X size: " + Arrays.toString(X.shape()));

        INDArray theta = Nd4j.zeros(3, 1);
        System.out.println(theta);
        // Expected 0.693
        System.out.println("Test Cost: " + computeCost(X, y, theta));

        INDArray grad = runGradientDescent(X, y, theta, 1, 1);
        // Expected -0.1, 012.0092, -11.2628
        System.out.println("Grad: " + grad);

        INDArray theta2 = Nd4j.create(new float[] { -24, 0.2f, 0.2f }, new int[] { 3, 1 });
        INDArray grad2 = runGradientDescent(X, y, theta2, 1, 1);
        // Expected 0.043, 2.566, 2.647
        System.out.println("Grad: " + grad2);

        //TODO 머신러닝 수업에 나온 것 외에 더 직접 dl4j가 제공하는 기능이 있으면 적용해 보자
        // 여기서는 matlab의 fminuc 함수를 안쓰고 경사하강법을 사용
//        INDArray gdTheta = runGradientDescent(X, y, theta, 0.01, 1000000);
        // 백만번 돌린 결과 -59.21,  0.47,  0.46 / 천만번 돌린 결과 -58.07,  0.45,  0.45
//        INDArray gdTheta = Nd4j.create(new float[] { -59.21f, 0.47f, 0.46f }, new int[] {3, 1});
        // fminuc의 값 -25.16127, 0.20623, 0.20147
        INDArray gdTheta = Nd4j.create(new float[] { -25.16127f, 0.20623f, 0.20147f }, new int[] { 3, 1 });
        System.out.println("Grad Theta: " + gdTheta);

        INDArray prob = sigmoid(Nd4j.create(new float[] { 1, 45, 85 }, new int[] { 1, 3 }).mmul(gdTheta));
        System.out.println("Predicted: " + prob);

        INDArray predicted = predict(gdTheta, X);
        System.out.println("Accuracy: " + predicted.eq(y).meanNumber());
    }

    private INDArray predict(INDArray theta, INDArray X) {
        INDArray result = sigmoid(X.mmul(theta));
        return result.gte(0.5);
    }

    private INDArray loadData() throws IOException {
        String filename = new ClassPathResource("ex2.txt").getFile().getPath();
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
