package kr.notforme.ml.ex1;

import static kr.notforme.ml.ex1.CostCalculator.computeCost;
import static kr.notforme.ml.ex1.GradientDescent.runGradientDescent;
import static kr.notforme.ml.support.PrintUtil.printDim;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

public class Ex1Main {
    public static void main(String[] args) throws IOException {
        new Ex1Main().runMain();
    }

    private void runMain() throws IOException {
        INDArray data = loadEx1();

        System.out.println("Data size: " + Arrays.toString(data.shape()));

        INDArray X = data.getColumn(0);
        INDArray y = data.getColumn(1);
        System.out.println("X size: " + Arrays.toString(X.shape()));
        System.out.println("y size: " + Arrays.toString(y.shape()));

        final int trainingSize = y.length();
        System.out.println("Training Size: " + trainingSize);

        INDArray ones = Nd4j.ones(trainingSize).transpose();
        System.out.println("Zeros size: " + Arrays.toString(ones.shape()));
        X = Nd4j.hstack(ones, X);
        System.out.println("X size: " + Arrays.toString(X.shape()));

        INDArray theta = Nd4j.zeros(2, 1);
        System.out.println(theta);
        System.out.println("Test Cost: " + computeCost(X, y, theta));

        INDArray theta2 = Nd4j.create(new float[] { -1, 2 }, new int[] { 2, 1 });
        System.out.println(theta2);
        printDim("theta2", theta2);
        System.out.println("Test Cost: " + computeCost(X, y, theta2));

        final int iterations = 1500;
        final double alpha = 0.01;
        theta = runGradientDescent(X, y, theta, alpha, iterations);


        //-3.6303  1.1664
        System.out.println(theta);
    }

    private INDArray loadEx1() throws IOException {
        String filename = new ClassPathResource("ex1.txt").getFile().getPath();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line = "";
        List<Double> vals = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            // use comma as separator
            String[] splited = line.split(",");
            vals.add(Double.valueOf(splited[0]));
            vals.add(Double.valueOf(splited[1]));
        }

        Double[] valArr = vals.toArray(new Double[vals.size()]);
        return Nd4j.create(ArrayUtils.toPrimitive(valArr), new int[] { vals.size() / 2, 2 });
    }
}
