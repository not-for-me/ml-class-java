package kr.notforme.ml;


import static kr.notforme.ml.support.PrintUtil.printDim;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PlaygroundMain {
    public static void main(String[] args) {
//        INDArray t1 = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
//        INDArray t2 = Nd4j.create(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
//        System.out.println("Matrix");
//        System.out.println(t1);
//        System.out.println(t2);
//        System.out.println("Calculation");
//        System.out.println(t1.mul(t2));
//        System.out.println(t1.muli(t2));
//        System.out.println("result");
//        System.out.println(t1);
//        System.out.println(t2);
//        System.out.println(t1.mmul(t2));
//        System.out.println(t1.mmuli(t2).toString());
//        System.out.println(t1.subi(t2).toString());
//        System.out.println(t1.divi(t2).toString());
//        System.out.println(t1.addi(t2).toString());
//
//        System.out.println(Nd4j.vstack(t1, t2));
//        System.out.println(Arrays.toString(Nd4j.hstack(t1, t2).shape()));


        INDArray x1 = Nd4j.create(new float[] { 1, 2, 1, 2,1,2 }, new int[] { 3, 2 });
        INDArray x2 = Nd4j.create(new float[] { 2, 2, 2, }, new int[] { 3, 1 });
//        System.out.println(x1.mulColumnVector(x2));
//        printDim("x1*x2",x1.mulColumnVector(x2));
//        System.out.println(x1.mulColumnVector(x2).sum(0));
//        printDim("sum(x1 * x2)",x1.mulColumnVector(x2).sum(0));
//        System.out.println(x1.mulColumnVector(x2).sum(1));
//        printDim("sum(x1 * x2) 2", x1.mulColumnVector(x2).sum(1));


    }




}
