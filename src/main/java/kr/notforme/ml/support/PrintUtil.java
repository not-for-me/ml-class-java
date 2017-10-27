package kr.notforme.ml.support;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

public class PrintUtil {
    public static void printDim(String label, INDArray a) {
        System.out.println(label + " Dim: " + Arrays.toString(a.shape()));
    }
}
