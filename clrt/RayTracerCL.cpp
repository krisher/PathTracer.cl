/*! 
 * \file RayTracer.cpp
 * \brief OpenCL based ray tracer main program
 *
 * http://developer.amd.com/GPU/ATISTREAMSDK/pages/TutorialOpenCL.aspx
 */
#include <utility>
/*
 * Enable C++ exceptions thrown from CL C++ bindings
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

// TODO: Linux only!
#include <GL/glx.h>

#include "PlatformInfo.h"
#include "RayTracerCL.h"
#include "Timing.h"

//Comment this out to disable debugging and timing info
//#define DEBUG

#define RAYTRACER_CL "ocl/raytracer.cl"
#ifdef DEBUG
#define CL_BUILD_OPTS "-Iocl -cl-nv-verbose -g"
#else /* Not DEBUG */
#define CL_BUILD_OPTS "-Iocl"
#endif /* DEBUG */

#define D2R(x) (x * M_PI/180.0f)

#define OUTBUFF_CL_PARAM 0
#define CAMERA_CL_PARAM 1
#define GEOM_CL_PARAM 2
#define GEOMCOUNT_CL_PARAM 3
#define IMWIDTH_CL_PARAM 4
#define IMHEIGHT_CL_PARAM 5
#define SAMPLERATE_CL_PARAM 6
#define MAXDEPTH_CL_PARAM 7
#define PROGRESSION_CL_PARAM 8
#define SEEDS_CL_PARAM 9

/*!
 * \brief Initializes RayTracer using the default CL Platform.
 */
RayTracerCL::RayTracerCL() :
    RayTracer() {
    init(*RayTracerCL::getDefaultPlatform());
}

RayTracerCL::RayTracerCL(cl::Platform const &platform) :
    RayTracer() {
    init(platform);
}

RayTracerCL::~RayTracerCL() {
    //TODO: Destroy context
    if (seedBufferCL)
        delete seedBufferCL;
}

void RayTracerCL::init(cl::Platform const &platform) {

    seedBufferCL = NULL;
    /*
     * Acquire a list of all of the devices available in the context.
     */
    std::vector<cl::Device> clDevices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);

    cl::Device device;
    createDefaultContext(&platform, context, device, glSharing);

    /*
     * Load CL files from disk, create a program from them, and build them for our selected device.
     */
    std::string raytracer_sources = *loadSourceFromFile(RAYTRACER_CL);
    cl::Program::Sources cl_source(1, std::make_pair(raytracer_sources.c_str(),
            raytracer_sources.length() + 1));
    cl::Program rtProgram(context, cl_source);

    /*
     * Compile the program for all devices in the list.
     * This will throw an exception if compilation fails, in which
     * case more specific information is recorded in the build log.
     *
     * TODO: Only compile for our selected target device.
     */

    try {
        rtProgram.build(clDevices, CL_BUILD_OPTS);
        std::cout << "Building Done!!!" << std::endl;
        std::string buildLog = rtProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG> (
                device);
        std::cout << "Build Log: " << std::endl << buildLog << std::endl;
    } catch (cl::Error buildErr) {
        std::string buildLog = rtProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG> (
                device);
        std::cerr << "Build Error: " << std::endl << buildLog << std::endl;
        throw(buildErr);
    }

    raytracer_kernel = cl::Kernel(rtProgram, "raytrace", NULL);

    /*
     * Compute the maximum work-group size for this kernel and the selected device.
     */
    size_t k_wg_size = getKernelMaxWGSize(&device, &raytracer_kernel);
    ndRangeSizes[0] = 32; //TODO: 32 might be too large
    ndRangeSizes[1] = k_wg_size / ndRangeSizes[0];

#ifdef DEBUG
    std::cout << "Kernel Max WG Size: " << k_wg_size << std::endl;
    std::cout << "Selected ND Range dimensions: [" << ndRangeSizes[0] << ", "
    << ndRangeSizes[1] << "]." << std::endl;
#endif /* DEBUG */
    /*
     * Create an OpenCL command queue
     */
    cmdQueue = cl::CommandQueue(context, clDevices[0], NULL /*properties...*/,
            NULL);

    glSharing = ::supportsGLSharing(clDevices[0]);

#ifdef DEBUG
    std::cout << "CL/GL Interoperability: " << (glSharing ? "Yes" : "No")
    << std::endl;
#endif

    /*
     * Allocate a constant buffer to pass the camera parameters.
     *
     * Note that the camera data is transferred in the rayTrace method, so no need to populate the data now.
     */
    cameraBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Camera));
    geomBufferSize = 0;
    sceneBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Sphere));

}

void RayTracerCL::updateSeedBuffer(uint width, uint height) {
    /*
     * Seed buffer
     */
    uint seed_count = width * height * 2;
    uint seeds[seed_count];
    for (uint i = 0; i < seed_count; ++i) {
        seeds[i] = rand();
        if (seeds[i] < 2) //?
            seeds[i] = 2;
    }

    //TODO: is the old buffer getting destroyed?
    if (seedBufferCL) {
        delete seedBufferCL;
    }
    seedBufferCL = new cl::Buffer(context, CL_MEM_READ_WRITE
            | CL_MEM_USE_HOST_PTR, sizeof(uint) * seed_count, (void *) seeds);
    try {
        raytracer_kernel.setArg(SEEDS_CL_PARAM, sizeof(cl_mem),
                &((*seedBufferCL)()));
    } catch (cl::Error err) {
        std::cout << "Err: " << err.err() << std::endl;
    }
}

void RayTracerCL::cameraChanged() {
    RayTracer::cameraChanged();
    camDirty = true;
}

void RayTracerCL::updateCLCamera() {
    gmtl::Vec3f view(0.0f, 0.0f, -1.0f);
    gmtl::Vec3f up(0.0f, 1.0f, 0.0f);

    view = viewMatrix * view;
    up = viewMatrix * up;
    gmtl::Vec3f right;
    gmtl::cross(right, view, up);

    Camera cameraCL;

    cameraCL.position.x = viewMatrix(0, 3);
    cameraCL.position.y = viewMatrix(1, 3);
    cameraCL.position.z = viewMatrix(2, 3);
    cameraCL.position.w = 0;

    cameraCL.up.x = up[0];
    cameraCL.up.y = up[1];
    cameraCL.up.z = up[2];
    cameraCL.up.w = 0.0f;

    cameraCL.right.x = right[0];
    cameraCL.right.y = right[1];
    cameraCL.right.z = right[2];
    cameraCL.right.w = 0.0f;

    view *= (float) ((width / 2.0) / tan(D2R(fovAngle) / 2.0));
    cameraCL.view.x = view[0];
    cameraCL.view.y = view[1];
    cameraCL.view.z = view[2];
    cameraCL.view.w = 0;

    //Synchronous Write...
    cmdQueue.enqueueWriteBuffer(cameraBufferCL, CL_TRUE, 0, sizeof(Camera),
            (void *) &cameraCL);
    raytracer_kernel.setArg(CAMERA_CL_PARAM, sizeof(cl_mem),
            &(cameraBufferCL()));
}

void RayTracerCL::rayTrace(cl_mem *buff, uint const width, uint const height,
        uint const progression) {
    if (width == 0 || height == 0)
        return;

    /*
     * Calculate the number (2D) of work items (one for each pixel), with some constraints
     * imposed by OpenCL:
     *
     * width is a multiple of 32 (workgroup width)
     * height is a multiple of the calculated wg. height
     */
    int wgMultipleWidth = ((width & 0x1F) == 0) ? width : ((width & 0xFFFFFFE0)
            + 0x20);
    int wgMutipleHeight = (int) ceil(height / (float) ndRangeSizes[1])
            * ndRangeSizes[1];

    /*
     * Update the camera
     */
    if (width != this->width || height != this->height) {
        camDirty = false;
        this->width = width;
        this->height = height;
        updateCLCamera();
        /*
         * Using a seed buffer that is a multiple of the wg size ensures global memory access can be coalesced.
         */
        updateSeedBuffer(wgMultipleWidth, wgMutipleHeight);
    } else if (camDirty) {
        camDirty = false;
        updateCLCamera();
    }

    if (sceneObjects.size() > 0 && geomBufferSize != (sizeof(Sphere)
            * sceneObjects.size()))

    {
        geomBufferSize = (sizeof(Sphere) * sceneObjects.size());
        sceneBufferCL = cl::Buffer(context, CL_MEM_READ_ONLY
                | CL_MEM_COPY_HOST_PTR, geomBufferSize,
                (void *) &(sceneObjects.front()));
        raytracer_kernel.setArg(GEOM_CL_PARAM, sizeof(cl_mem),
                &(sceneBufferCL()));
        raytracer_kernel.setArg(GEOMCOUNT_CL_PARAM, (uint) sceneObjects.size());
    } else {
        //TODO: check whether the scene content has changed but #elements did not
    }

    /*
     * Transfer the buffer and image size info.
     */
    raytracer_kernel.setArg(OUTBUFF_CL_PARAM, sizeof(cl_mem), buff);
    raytracer_kernel.setArg(IMWIDTH_CL_PARAM, width);
    raytracer_kernel.setArg(IMHEIGHT_CL_PARAM, height);
    raytracer_kernel.setArg(SAMPLERATE_CL_PARAM, sampleRate);
    raytracer_kernel.setArg(MAXDEPTH_CL_PARAM, maxPathDepth);
    raytracer_kernel.setArg(PROGRESSION_CL_PARAM, progression);
    //TODO: use KernelFunctor -- fixed NDRange...?
    try {
        /*
         * Random Notes:
         *
         * NVidia QuadroFX 5800 supports 512 work-items per wg (32 x 16), however the ray tracer code uses too many registers
         * for 512 items (register file is shared among items in the same group).  6x32 is the minimum value to hide certain latencies
         * when accessing memory or read-after-write registers according to their OpenCL best-practices guide.
         *
         * NVidia GTX 460 supports 1024 work-items per wg (32x32), but due to register use this maxes out at 32x16 (512).
         *
         * Number of suggested work items is queried from OpenCL driver to get a good value based on hardware capabilities and
         * the number of regs actually used by the kernel.
         */
        cmdQueue.enqueueNDRangeKernel(raytracer_kernel, cl::NullRange,
                cl::NDRange(wgMultipleWidth, wgMutipleHeight), cl::NDRange(
                        ndRangeSizes[0], ndRangeSizes[1]));
        //cmdQueue.finish();
#ifdef DEBUG
        //		std::cout << "CL Render Time: " << timeElapsed(startTicks) << "s"
        //				<< std::endl;
#endif /* DEBUG */
    } catch (cl::Error err) {
        std::cerr << "Error submitting kernel for execution: " << err.err()
                << std::endl;
        std::cerr << "Global x: " << width << ", Global y: " << height
                << std::endl;
        throw(err);
    }

    //Not sure why this was here, forces update of camera and random seed data every frame.
    //camDirty = true;
}

const cl::Context RayTracerCL::getCLContext() {
    return this->context;
}

void RayTracerCL::createDefaultContext(const cl::Platform *platform,
        cl::Context &context, cl::Device &device, bool &glSharing) {
    cl_context_properties cps[7] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) (*platform)(),
            // TODO: Linux only!
            CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
            0 };
    /*
     * Acquire a list of all of the devices available in the context.
     */
    std::vector<cl::Device> clDevices;
    platform->getDevices(CL_DEVICE_TYPE_ALL, &clDevices);

#ifdef DEBUG
    std::cout << std::endl << "OpenCL Devices: " << std::endl;
    std::vector<cl::Device>::iterator devItr;
    for (devItr = clDevices.begin(); devItr != clDevices.end(); ++devItr) {
        ::printDeviceInfo(&(*devItr));
    }
    size_t dev_idx = 0;
    if (clDevices.size() > 1) {
        std::cout << std::endl << "Select a devices number (0 - "
        << clDevices.size() - 1 << "): ";
        std::cin >> dev_idx;
        if (dev_idx >= clDevices.size() || dev_idx < 0)
        dev_idx = 0;
    }
    std::cout << std::endl << "Selected Device: " << std::endl;
    device = clDevices[dev_idx];
    ::printDeviceInfo(&device);
    clDevices.clear();
    clDevices.push_back(device);
    context = cl::Context(clDevices, cps, NULL, NULL, NULL);
#else /* Not DEBUG */
    try {
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, NULL);
    } catch (cl::Error err) {
        std::cout
                << "Unable to create context for GPU device, falling back to CPU device."
                << std::endl;
        /*
         * CPU is rather unlikely to support GL sharing...
         */
        cps[2] = 0;
        context = cl::Context(CL_DEVICE_TYPE_CPU, cps, NULL, NULL, NULL);
        glSharing = false;
    }
    device = context.getInfo<CL_CONTEXT_DEVICES> ()[0];
#endif /* DEBUG */
}

cl::Platform *RayTracerCL::getDefaultPlatform() {
    /*
     * Get a list of the available platforms.
     */
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (cl::Error err) {
        std::cerr << "Error getting platform IDs: " << err.what() << std::endl;
        throw(err);
    }

    // FIXME: clean up un-used cl::Platform objects?
    if (platforms.size() > 0) {
        std::vector<cl::Platform>::iterator i;
#ifdef DEBUG
        std::cout << "Available Platforms: " << std::endl;
        unsigned int idx = 0;
        for (i = platforms.begin(); i != platforms.end(); ++i) {
            std::cout << "--Platform " << idx << ": " << std::endl;
            ::printPlatformInfo(&(*i));
            ++idx;
        }
        unsigned int platform = 0;
        if (platforms.size() > 1) {
            std::cout << std::endl << "Enter a platform number: ";
            std::cin >> platform;
            if (platform >= platforms.size() || platform < 0)
            platform = 0;
        }
        std::cout << std::endl << "Selected Platform: " << std::endl;
        ::printPlatformInfo(&platforms[platform]);
        return new cl::Platform(platforms[platform]);
#else /* Not debug, choose first platform by default. */
        return new cl::Platform(platforms[0]);
#endif /* DEBUG */
    }
    // TODO: throw exception if out not initialized...
    return NULL;
}

/*!
 * Returns a cl::Program::Sources vector containing the contents of the specified file.
 */
std::string *RayTracerCL::loadSourceFromFile(const char *filename) {
    std::ifstream kernel_file(filename);
    if (!kernel_file.is_open()) {
#ifdef DEBUG
        std::cerr << "Unable to open file: " << filename << std::endl;
#endif
        return NULL;
    }
    return new std::string(std::istreambuf_iterator<char>(kernel_file),
            (std::istreambuf_iterator<char>()));
}

