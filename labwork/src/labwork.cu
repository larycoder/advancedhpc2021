#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <iostream>

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
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
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
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
        printf("labwork 1 openMP elllapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            std::cout << "blockSize: "; std::cin >> blockSize;
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
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            printf("labwork 5 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
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
    for (int i = 0; i < nDevices; i++){
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
        printf("Bandwidth: %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

}

__global__ void grayScale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int buffer =  ((int)input[tid].x + (int)input[tid].y + (int)input[tid].z) / 3;
    output[tid].x = (char)buffer;
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
    outputImage = (char*) malloc(pixelCount);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}


__global__ void grayScaleMultiDim(uchar3 *input, uchar3 *output, int w, int h) {
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;
    if(tidX > w || tidY > h) return;
    int tid = tidX + tidY * w;
    int buffer =  ((int)input[tid].x + (int)input[tid].y + (int)input[tid].z) / 3;
    output[tid].x = (char)buffer;
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
    dim3 bSize = dim3(32, 32);

    int dH = inputImage->height / 32 + (inputImage->height % 32 ? 1:0);
    int dW = inputImage->width / 32 + (inputImage->width % 32 ? 1:0);
    dim3 gridSize = dim3(dW, dH);
    grayScaleMultiDim<<<gridSize, bSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    outputImage = (char*) malloc(pixelCount);
    cudaMemcpy(outputImage, devOutput, pixelCount, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);
    cudaFree(devOutput);
}


int kernel[7][7] = {
   {0,0,1,2,1,0,0},
   {0,3,13,22,13,3,0},
   {1,13,59,97,59,13,1},
   {2,22,97,159,97,22,2},
   {1,13,59,97,59,13,1},
   {0,3,13,22,13,3,0},
   {0,0,1,2,1,0,0}
};


void Labwork::labwork5_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // copy input array to output array
    for(int i = 0; i < pixelCount*3; i++) {
        outputImage[i] = inputImage->buffer[i];
    }

    // kernel sum
    int kSum = 0;
    for(int i = 0; i < 7; i++)
            for(int j = 0; j < 7; j++) kSum += kernel[i][j];

    // blur image
    for (int row = 3; row < inputImage->height-3; row++) {
        for (int col = 3; col < inputImage->width-3; col++) {
            int sR = 0, sG = 0, sB = 0;
            // convolution
            for(int x = 0; x < 7; x++) {
                for(int y = 0; y < 7; y++) {
                    int iid = (row - 3 + x) * inputImage->width + (col - 3 + y);
                    sR += (int) inputImage->buffer[iid*3] * kernel[x][y];
                    sG += (int) inputImage->buffer[iid*3+1] * kernel[x][y];
                    sB += (int) inputImage->buffer[iid*3+2] * kernel[x][y];
                }
            }
            int oid = row * inputImage->width + col;
            outputImage[oid*3] = (char) (sR/kSum);
            outputImage[oid*3+1] = (char) (sG/kSum);
            outputImage[oid*3+2] = (char) (sB/kSum);
        }
    }
}


__global__ void gausBlur(uchar3* input, uchar3* output, int w, int h, int kSum) {
    // kernel
    int kernel[7][7] = {
            {0,0,1,2,1,0,0},
            {0,3,13,22,13,3,0},
            {1,13,59,97,59,13,1},
            {2,22,97,159,97,22,2},
            {1,13,59,97,59,13,1},
            {0,3,13,22,13,3,0},
            {0,0,1,2,1,0,0}
    };

    // blur image
    int sR = 0, sG = 0, sB = 0;

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if((row < 3 || row > h - 3) || (col < 3 || col > w)) return;

    // convolution
    for(int x = 0; x < 7; x++) {
        for(int y = 0; y < 7; y++) {
            int iid = (row - 3 + x) * w + (col - 3 + y);
            sR += (int) input[iid].x * kernel[x][y];
            sG += (int) input[iid].y * kernel[x][y];
            sB += (int) input[iid].z * kernel[x][y];
        }
    }
    int oid = row * w + col;
    output[oid].x = (char) (sR/kSum);
    output[oid].y = (char) (sG/kSum);
    output[oid].z = (char) (sB/kSum);
}


void Labwork::labwork5_GPU() {
    // allocate cuda mem
    uchar3 *devInput;
    uchar3 *devOutput;
    int pixelCount = inputImage->width * inputImage->height;
    cudaMalloc(&devInput, pixelCount*3);
    cudaMalloc(&devOutput, pixelCount*3);
    cudaMemcpy(devInput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
    //cudaMemcpy(devOutput, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);

    // kernel sum
    int kSum = 0;
    for(int i = 0; i < 7; i++)
            for(int j = 0; j < 7; j++) kSum += kernel[i][j];

    // Processing
    dim3 bS = dim3(16, 16);
    int dH = inputImage->height / 16 + (inputImage->height % 16 ? 1:0);
    int dW = inputImage->width / 16 + (inputImage->width % 16 ? 1:0);
    dim3 gS = dim3(dH, dW);
    printf("Block Size: %d, Grid Size: %d\n", 32*32, dH*dW);

    gausBlur<<<gS, bS>>>(devInput, devOutput, inputImage->width, inputImage->height, kSum);
    //grayScaleMultiDim<<<gS, bS>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy cuda mem grom GPU to CPU
    outputImage = (char*) malloc(pixelCount*3);
    cudaMemcpy(outputImage, devOutput, pixelCount*3, cudaMemcpyDeviceToHost);
    cudaFree(devInput);
    cudaFree(devOutput);
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























