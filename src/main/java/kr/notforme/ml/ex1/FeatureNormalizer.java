package kr.notforme.ml.ex1;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FeatureNormalizer {
    public static Norm norm(INDArray X) {
        INDArray mean = X.mean(0);
        INDArray sigma = X.std(0);

        INDArray result = X.subRowVector(mean).divRowVector(sigma);

        return new Norm(mean, sigma, result);
    }

    public static class Norm {
        private final INDArray mean;
        private final INDArray sigma;
        private final INDArray result;

        public Norm(INDArray mean, INDArray sigma, INDArray result) {
            this.mean = mean;
            this.sigma = sigma;
            this.result = result;
        }

        public INDArray getNormValue(INDArray input) {
            return input.subRowVector(mean).divRowVector(sigma);
        }

        public INDArray getMean() {
            return mean;
        }

        public INDArray getSigma() {
            return sigma;
        }

        public INDArray getResult() {
            return result;
        }
    }
}
