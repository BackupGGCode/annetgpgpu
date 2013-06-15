#ifndef ANFUNCTORS_H_
#define ANFUNCTORS_H_

#include "../math/Functions.h"


struct sAXpY_functor { // Y <- A * X + Y
    const float a;

    sAXpY_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x, const float& y) const {
		return a * x + y;
	}
};

struct sAX_functor { // Y <- A * X
    const float a;

    sAX_functor(float _a) : a(_a) {}

    __host__ __device__
	float operator()(const float& x) const {
		return a * x;
	}
};

struct sAXmY_functor { // Y <- A * (X - Y)
	const float a;

	sAXmY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x, const float& y) const { 
		return a * (x - y);
	}
};

struct sXmAmY_functor { // Y <- X - (A - Y)
	const float a;

	sXmAmY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x, const float& y) const { 
		return x - (a - y);
	}
};

struct spowAmXpY_functor { // Y <- (A-X)^2 + Y
	const float a;

	spowAmXpY_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float& x, const float& y) const { 
		return pow(a-x, 2) + y;
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct sm13bubble_functor {
	float fSigmaT;
	sm13bubble_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(const float& dist) const {
		return ANN::fcn_bubble_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13gaussian_functor {
	float fSigmaT;
	sm13gaussian_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(const float& dist) const {
		return ANN::fcn_gaussian_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13cut_gaussian_functor {
	float fSigmaT;
	sm13cut_gaussian_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(const float& dist) const {
		return ANN::fcn_cutgaussian_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13mexican_functor {
	float fSigmaT;
	sm13mexican_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(const float& dist) const {
		return ANN::fcn_mexican_nhood(sqrt(dist), fSigmaT);
	}
};

struct sm13epanechicov_functor {
	float fSigmaT;
	sm13epanechicov_functor(const float &sigmaT) : fSigmaT(sigmaT)	{}

	__host__ __device__
	float operator()(const float& dist) const {
		return ANN::fcn_epanechicov_nhood(sqrt(dist), fSigmaT);
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
struct hebbian_functor {
	float fLearningRate;
	float fInput;

	hebbian_functor(const float &learning_rate, const float &input) :
		fLearningRate(learning_rate), fInput(input) {}

	__host__ __device__
	float operator()(const float& fWeight, const float& fInfluence) const {
		return fWeight + (fInfluence*fLearningRate*(fInput-fWeight) );
	}
};
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
#if __CUDA_ARCH__ >= 200
typedef float (*external_device_func_t) (const float&, const float&);
struct sm20distance_functor {
	float fSigmaT;
	external_device_func_t* m_pfunc;
	sm20distance_functor(const float &sigmaT, external_device_func_t* pfunc) : 
		fSigmaT(sigmaT), m_pfunc(pfunc) {}

    __host__ __device__
	float operator()(const float& dist) const {
		return (**m_pfunc)(sqrt(dist), fSigmaT);
	}
};
#endif

#endif