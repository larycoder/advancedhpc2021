#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <iostream>
#include <math.h>

#define ACTIVE_THREADS 4

int blockSize = 64;

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename, inputFilename1, inputFilename2;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum == 6) {
        float threshold = atoi(argv[2]);
        inputFilename1 = std::string(argv[3]);
        inputFilename2 = std::string(argv[4]);
        labwork.setThreshold(threshold);
        labwork.loadInputImage(inputFilename1);
        labwork.loadInputImage2(inputFilename2);
    } else if (lwNum != 2) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork %d CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            printf("labwork %d openMP elllapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            std::cout << "blockSize: ";
            std::cin >> blockSize;
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            printf("labwork %d CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork5_GPU(false);
            labwork.saveOutputImage("labwork5-gpu-out-no-shared.jpg");
            printf("labwork %d GPU no shared mem ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork5_GPU(true);
            labwork.saveOutputImage("labwork5-gpu-out-shared.jpg");
            printf("labwork %d GPU shared mem ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 6:
            labwork.labwork6_GPU('b');
            labwork.saveOutputImage("labwork6-gpu-out-binary.jpg");
            printf("labwork %d GPU binary ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());

            timer.start();
            labwork.labwork6_GPU('n');
            labwork.saveOutputImage("labwork6-gpu-out-brightness.jpg");
            printf("labwork %d GPU brighness ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());

            timer.start();
            labwork.labwork6_GPU('d');
            labwork.saveOutputImage("labwork6-gpu-out-blending.jpg");
            printf("labwork %d GPU blending ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::loadInputImage2(std::string inputFileName) {
    inputImage2 = jpegLoader.load(inputFileName);
}

void Labwork::setThreshold(float threshold) {
    this->threshold = threshold;
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // do something here
# pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++) {
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        // Device name + core info (clock rate, core counts, multiprocessor count, warp size)
        // Memory info (clock rate, bus width, bandwidth)
        printf("Device name: %s\n\n", prop.name);
        printf("Core info:\n");
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("Warp Size: %d\n\n", prop.warpSize);
        printf("Memory info:\n");
        printf("Clock Rate: %d\n", prop.memoryClockRate);
        printf("Bus Width: %d\n", prop.memoryBusWidth);
        printf("Bandwidth: %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

}

__global__ void grayScale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int buffer = ((int) input[tid].x + (int) input[tid].y + (int) input[tid].z) / 3;
    output[tid].x = (char) buffer;
    output[tid].y = output[tid].z = output[tid].x;
}

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    // inputImage struct: width, height, buffer
    int pixelCount = inputImage->width * inputImage->height * 3;

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount);
    cudaMalloc(&devOutput, pixelCount);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    int numBlock = pixelCount / (blockSize * 3);
    grayScale<<<numBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    outputImage = (char *) malloc(pixelCount);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}


__global__ void grayScaleMultiDim(uchar3 *input, uchar3 *output, int w, int h) {
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidX >= w || tidY >= h) return;
    int tid = tidX * h + tidY;
    int buffer = ((int) input[tid].x + (int) input[tid].y + (int) input[tid].z) / 3;
    output[tid].x = (char) buffer;
    output[tid].y = output[tid].z = output[tid].x;
}


void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    // inputImage struct: width, height, buffer
    int pixelCount = inputImage->width * inputImage->height * 3;

    // Allocate CUDA memory
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount);
    cudaMalloc(&devOutput, pixelCount);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    dim3 bSize = dim3(16, 16);

    int dH = inputImage->height / 16 + (inputImage->height % 16 ? 1 : 0);
    int dW = inputImage->width / 16 + (inputImage->width % 16 ? 1 : 0);
    dim3 gridSize = dim3(dW, dH);

    std::cout << "Block Size: " << 16 * 16 << std::endl;
    std::cout << "Grid Size: " << dW * dH << std::endl;

    grayScaleMultiDim<<<gridSize, bSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    outputImage = (char *) malloc(pixelCount);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}


int kernel[7][7] = {
        {0, 0,  1,  2,   1,  0,  0},
        {0, 3,  13, 22,  13, 3,  0},
        {1, 13, 59, 97,  59, 13, 1},
        {2, 22, 97, 159, 97, 22, 2},
        {1, 13, 59, 97,  59, 13, 1},
        {0, 3,  13, 22,  13, 3,  0},
        {0, 0,  1,  2,   1,  0,  0}
};


void Labwork::labwork5_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // copy input array to output array
    for (int i = 0; i < pixelCount * 3; i++) {
        outputImage[i] = inputImage->buffer[i];
    }

    // kernel sum
    int kSum = 0;
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) kSum += kernel[i][j];

    // blur image
    for (int row = 3; row < inputImage->height - 3; row++) {
        for (int col = 3; col < inputImage->width - 3; col++) {
            int sR = 0, sG = 0, sB = 0;
            // convolution
            for (int x = 0; x < 7; x++) {
                for (int y = 0; y < 7; y++) {
                    int iid = (row - 3 + x) * inputImage->width + (col - 3 + y);
                    sR += (int) inputImage->buffer[iid * 3] * kernel[x][y];
                    sG += (int) inputImage->buffer[iid * 3 + 1] * kernel[x][y];
                    sB += (int) inputImage->buffer[iid * 3 + 2] * kernel[x][y];
                }
            }
            int oid = row * inputImage->width + col;
            outputImage[oid * 3] = (char) (sR / kSum);
            outputImage[oid * 3 + 1] = (char) (sG / kSum);
            outputImage[oid * 3 + 2] = (char) (sB / kSum);
        }
    }
}


__global__ void gausBlurNoSharedMem(uchar3 *input, uchar3 *output, int w, int h, int kSum) {
    // kernel
    int kernel[7][7] = {
            {0, 0,  1,  2,   1,  0,  0},
            {0, 3,  13, 22,  13, 3,  0},
            {1, 13, 59, 97,  59, 13, 1},
            {2, 22, 97, 159, 97, 22, 2},
            {1, 13, 59, 97,  59, 13, 1},
            {0, 3,  13, 22,  13, 3,  0},
            {0, 0,  1,  2,   1,  0,  0}
    };

    // blur image
    int sR = 0, sG = 0, sB = 0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((row < 3 || row >= h - 3) || (col < 3 || col >= w - 3)) return;

    // convolution
    for (int x = 0; x < 7; x++) {
        for (int y = 0; y < 7; y++) {
            int iid = (row - 3 + x) * w + (col - 3 + y);
            sR += (int) input[iid].x * kernel[x][y];
            sG += (int) input[iid].y * kernel[x][y];
            sB += (int) input[iid].z * kernel[x][y];
        }
    }
    int oid = row * w + col;
    output[oid].x = (char) (sR / kSum);
    output[oid].y = (char) (sG / kSum);
    output[oid].z = (char) (sB / kSum);
}


__global__ void gausBlurSharedMem(uchar3 *input, uchar3 *output, int w, int h, int kSum, int *dKernel) {
    // kernel
    __shared__ int kernel[7][7];
    if (threadIdx.x < 7 && threadIdx.y < 7) {
        kernel[threadIdx.x][threadIdx.y] = dKernel[threadIdx.x * 7 + threadIdx.y];
    }
    __syncthreads();

    // blur image
    int sR = 0, sG = 0, sB = 0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if ((row < 3 || row >= h - 3) || (col < 3 || col >= w - 3)) return;

    // convolution
    for (int x = 0; x < 7; x++) {
        for (int y = 0; y < 7; y++) {
            int iid = (row - 3 + x) * w + (col - 3 + y);
            uchar3 rI = input[iid];
            sR += (int) rI.x * kernel[x][y];
            sG += (int) rI.y * kernel[x][y];
            sB += (int) rI.z * kernel[x][y];
        }
    }
    int oid = row * w + col;
    uchar3 rO;
    rO.x = (char) (sR / kSum);
    rO.y = (char) (sG / kSum);
    rO.z = (char) (sB / kSum);
    output[oid] = rO;
}


void Labwork::labwork5_GPU(bool shareMem) {
    // allocate cuda mem
    uchar3 *devInput;
    uchar3 *devOutput;
    int *dKernel;

    int hKernel[7 * 7] = {
            0, 0, 1, 2, 1, 0, 0,
            0, 3, 13, 22, 13, 3, 0,
            1, 13, 59, 97, 59, 13, 1,
            2, 22, 97, 159, 97, 22, 2,
            1, 13, 59, 97, 59, 13, 1,
            0, 3, 13, 22, 13, 3, 0,
            0, 0, 1, 2, 1, 0, 0
    };
    int pixelCount = inputImage->width * inputImage->height;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    cudaMalloc(&dKernel, 7 * 7 * 4);
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, hKernel, 7 * 7 * 4, cudaMemcpyHostToDevice);

    // kernel sum
    int kSum = 0;
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) kSum += kernel[i][j];

    // Processing
    dim3 bS = dim3(16, 16);
    int dH = inputImage->height / 16 + (inputImage->height % 16 ? 1 : 0);
    int dW = inputImage->width / 16 + (inputImage->width % 16 ? 1 : 0);
    dim3 gS = dim3(dH, dW);
    printf("Block Size: %d, Grid Size: %d\n", 32 * 32, dH * dW);

    if (!shareMem)
        gausBlurNoSharedMem<<<gS, bS>>>(devInput, devOutput, inputImage->width, inputImage->height, kSum);
    else
        gausBlurSharedMem<<<gS, bS>>>(devInput, devOutput, inputImage->width, inputImage->height, kSum, dKernel);

    // Copy cuda mem grom GPU to CPU
    outputImage = (char *) malloc(pixelCount * 3);
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devOutput);
}


__global__ void convertImageToBinary(
        uchar3 *in, uchar3 *out, float threshold, int w, int h) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iIdx = ix * h + iy;
    if (ix >= w || iy >= h) return;

    uchar3 input = in[iIdx];
    float iValue = ((float) input.x + (float) input.y + (float) input.z) / 3;
    unsigned char oValue = (iValue > threshold ? 255 : 0);

    uchar3 output;
    output.x = output.y = output.z = oValue;
    out[iIdx] = output;
}


__global__ void increaseImageBrightness(
        uchar3 *in, uchar3 *out, char threshold, int w, int h) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iIdx = ix + iy * w;
    if (ix >= w || iy >= h) return;

    uchar3 input = in[iIdx];
    uchar3 output;

    int newRed = (int) input.x + (int) threshold;
    int newGreen = (int) input.y + (int) threshold;
    int newBlue = (int) input.z + (int) threshold;

    output.x = (char) (newRed > 255 ? 255 : newRed);
    output.y = (char) (newGreen > 255 ? 255 : newGreen);
    output.z = (char) (newBlue > 255 ? 255 : newBlue);
    out[iIdx] = output;
}


__global__ void blendingImage(
        uchar3 *in1, uchar3 *in2, uchar3 *out,
        int w1, int w2, int w, int h) {
    int xId = threadIdx.x + blockDim.x * blockIdx.x;
    int yId = threadIdx.y + blockDim.y * blockIdx.y;
    if (xId >= w || yId >= h) return;

    int index = xId * h + yId;
    uchar3 rIn1 = in1[index], rIn2 = in2[index];

    uchar3 rOut;
    rOut.x = ((int) rIn1.x * w1 + (int) rIn2.x * w2) / (w1 + w2);
    rOut.y = ((int) rIn1.y * w1 + (int) rIn2.y * w2) / (w1 + w2);
    rOut.z = ((int) rIn1.z * w1 + (int) rIn2.z * w2) / (w1 + w2);
    out[index] = rOut;
}


void Labwork::labwork6_GPU(char type) {
    int pixelCount = inputImage->width * inputImage->height;

    uchar3 *in1, *in2, *out;
    cudaMalloc(&in1, pixelCount * 3);
    cudaMalloc(&in2, pixelCount * 3);
    cudaMalloc(&out, pixelCount * 3);

    cudaMemcpy(in1, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(in2, inputImage2->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    dim3 bS = dim3(16, 16);
    int dH = inputImage->height / 16 + (inputImage->height % 16 ? 1 : 0);
    int dW = inputImage->width / 16 + (inputImage->width % 16 ? 1 : 0);
    dim3 gS = dim3(dW, dH);

    if (type == 'b') {
        // binary
        convertImageToBinary<<<gS, bS>>>(in1, out, threshold, inputImage->width, inputImage->height);
        std::cout << "signal 2" << std::endl;
    } else if (type == 'n') {
        // brightness
        increaseImageBrightness<<<gS, bS>>>(in1, out, 10, inputImage->width, inputImage->height);
    } else if (type == 'd') {
        // blending
        // width and height will be minimum of both 2 input image
        int width1 = inputImage->width, width2 = inputImage2->width;
        int height1 = inputImage->height, height2 = inputImage2->height;
        int width = width1 > width2 ? width2 : width1;
        int height = height1 > height2 ? height2 : height1;

        dH = height / 16 + (height % 16 ? 1 : 0);
        dW = width / 16 + (width % 16 ? 1 : 0);
        gS = dim3(dW, dH);
        blendingImage<<<gS, bS>>>(in1, in2, out, 20, 10, width, height);
    }

    outputImage = (char *) malloc(pixelCount * 3);
    cudaMemcpy(outputImage, out, pixelCount * 3, cudaMemcpyDeviceToHost);
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(out);
}

namespace LW7 {
    void getLastCudaError(std::string err) {
        cudaError errCheck = cudaDeviceSynchronize();
        if (cudaSuccess != errCheck) {
            std::cout << err << std::endl;
            std::cout << cudaGetErrorString(errCheck) << std::endl;
            exit(-1);
        }
    }


    namespace REDUCE {
        template<typename FUNC>
        __global__
        void reduce_block(uchar3 *out, uchar3 *in, FUNC functor, int size) {
            extern __shared__ uchar3 cache[];
            int local = threadIdx.x;
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= size) return;
            cache[local] = in[tid];
            __syncthreads();
            for (int s = 1; s < blockDim.x; s *= 2) {
                if (local % (s * 2) == 0) {
                    cache[local] = functor(cache[local], cache[local + s], !(tid + s >= size));
                }
                __syncthreads();
            }

            if (local == 0) out[blockIdx.x] = cache[0];
        }

        struct maxFunctor {
            __device__
            uchar3 operator()(const uchar3 &first, const uchar3 &second, const bool bounded) const {
                if (!bounded) return first;
                return {
                        (first.x > second.x) ? first.x : second.x,
                        (first.y > second.y) ? first.y : second.y,
                        (first.z > second.z) ? first.z : second.z,
                };
            }
        };

        struct minFunctor {
            __device__
            uchar3 operator()(const uchar3 &first, const uchar3 &second, const bool bounded) const {
                if (!bounded) return first;
                return {
                        (first.x < second.x) ? first.x : second.x,
                        (first.y < second.y) ? first.y : second.y,
                        (first.z < second.z) ? first.z : second.z,
                };
            }
        };
    }

    template<typename T>
    struct DeviceData {
        int size;
        T *data;
        const int threadSize = 512;
        int blockSize;

        DeviceData(T *data, int size) {
            this->size = size;
            const int realSize = sizeof(T) * this->size;
            cudaMalloc(&this->data, realSize);
            getLastCudaError("Could not allocate a DeviceData...");
            cudaMemcpy(this->data, data, realSize, cudaMemcpyHostToDevice);
            getLastCudaError("Could not copy data to DeviceData...");
            getBlockSize();
        }

        DeviceData(int size) {
            const int realSize = sizeof(T) * size;
            cudaMalloc(&this->data, realSize);
            getLastCudaError("Could not allocate a DeviceData...");
            this->size = size;
            getBlockSize();
        }

        ~DeviceData() {
            cudaFree(data);
            getLastCudaError("Could not free DeviceData...");
        }

        void getBlockSize() {
            blockSize = (int) (size + threadSize - 1) / threadSize;
        }

        __device__
        T operator[](const int &idx) const {
            return data[idx];
        }

        __device__
        T &at(int idx) {
            return data[idx];
        }

        __host__
        void copyToHost(T *holder) {
            cudaMemcpy(holder, data, size * sizeof(T), cudaMemcpyDeviceToHost);
        }
    };

    __global__
    void convertToGrayScale(uchar3 *out, uchar3 *in, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= size) return;

        uchar3 pixel = in[tid];
        unsigned char mid = ((int) pixel.x + (int) pixel.y + (int) pixel.z) / 3;
        out[tid] = {mid, mid, mid};
    }

    __global__
    void stretchMap(uchar3 *out, uchar3 *in, uchar3 *min, uchar3 *max, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid >= size) return;

        uchar3 rOut = out[tid], rIn = in[tid], rMin = min[0], rMax = max[0];
        float rangeX = (float) rMax.x - (float) rMin.x;
        float rangeY = (float) rMax.y - (float) rMin.y;
        float rangeZ = (float) rMax.z - (float) rMin.z;
        rOut.x = 255.0 * ((float) rIn.x - (float) rMin.x) / rangeX;
        rOut.y = 255.0 * ((float) rIn.y - (float) rMin.y) / rangeY;
        rOut.z = 255.0 * ((float) rIn.z - (float) rMin.z) / rangeZ;
        out[tid] = rOut;
    }

    template<typename FUNC>
    __host__
    void reduce(
            uchar3 *out, uchar3 *in, int size, FUNC functor) {
        if (size == 1) return;
        const int threads = 512;
        const int blocks = (size + threads - 1) / threads;

        REDUCE::reduce_block<FUNC><<<blocks, threads, sizeof(uchar3) * threads>>>(out, in, functor, size);
        getLastCudaError("Reduce framework go wrong...");
        reduce<FUNC>(out, out, blocks, functor);
    }
}

void Labwork::labwork7_GPU() {
    using namespace LW7;
    const int imageSize = inputImage->width * inputImage->height;
    DeviceData<uchar3> image((uchar3 *) inputImage->buffer, imageSize);
    DeviceData<uchar3> max(image.size), min(image.size), gray(image.size);
    DeviceData<uchar3> out(image.size);

    convertToGrayScale<<<image.blockSize, image.threadSize>>>(gray.data, image.data, image.size);
    reduce<REDUCE::maxFunctor>(max.data, gray.data, imageSize, REDUCE::maxFunctor());
    reduce<REDUCE::minFunctor>(min.data, gray.data, imageSize, REDUCE::minFunctor());
    stretchMap<<<out.blockSize, out.threadSize>>>(out.data, gray.data, min.data, max.data, imageSize);

    outputImage = (char *) malloc(imageSize * 3);
    out.copyToHost((uchar3 *) outputImage);
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {
}


























