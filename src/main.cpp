/*
 * Author: Joshua Salmons
 * Course: Parallel Computing Techniques CSCN73000
 * Date: November 19th 2025
 */

#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0, cols = 0, channels = 0;
    vector<uchar> flat_image;

    if (rank == 0) {
        Mat image = imread("input.jpg", IMREAD_COLOR);

        if (image.empty()) {
            cerr << "Error loading image.\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        rows = image.rows;
        cols = image.cols;
        channels = image.channels();
        flat_image.assign(image.data, image.data + image.total() * channels);
    }

    // Broadcast image metadata from master to all processes
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate how many pixels each process will handle
    int total_pixels = rows * cols * channels;
    int chunk_size = total_pixels / size;
    vector<uchar> local_chunk(chunk_size);

    // Distribute pixel data evenly across all processes
    MPI_Scatter(flat_image.data(), chunk_size, MPI_UNSIGNED_CHAR,
                local_chunk.data(), chunk_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Each process inverts the colors of its assigned pixels
    for (int i = 0; i < chunk_size; ++i) {
        local_chunk[i] = 255 - local_chunk[i];
    }

    // Calculate the average pixel value for this process's chunk
    double local_sum = 0.0;
    for (int i = 0; i < chunk_size; ++i) {
        local_sum += local_chunk[i];
    }
    double local_average = local_sum / chunk_size;

    // Prepare buffer on master process to receive all inverted pixels
    vector<uchar> inverted_image;
    if (rank == 0) {
        inverted_image.resize(total_pixels);
    }

    // Gather all inverted pixel chunks back to the master process
    MPI_Gather(local_chunk.data(), chunk_size, MPI_UNSIGNED_CHAR,
              inverted_image.data(), chunk_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // Prepare buffer on master to receive all local averages
    vector<double> all_averages;
    if (rank == 0) {
        all_averages.resize(size);
    }

    // Gather local averages from all processes to master
    MPI_Gather(&local_average, 1, MPI_DOUBLE,
            all_averages.data(), 1, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

	// Master process displays results and saves the inverted image
    if (rank == 0) {
        // Display the average pixel value calculated by each process
        for (int i = 0; i < size; ++i) {
            cout << "Process " << i << " average: " << all_averages[i] << endl;
        }
        // Reconstruct the image from the inverted pixel data
        Mat output_image(rows, cols, CV_8UC3, inverted_image.data());
        imwrite("output.jpg", output_image);
    }
    MPI_Finalize();
    return 0;
}