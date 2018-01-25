import cython
import numpy as np
cimport numpy as np

cdef extern from "c_NumericLayers.h":
	cdef extern void c_mySolvePnP(double * input_array1, double * input_array2, double * input_array3, 
		double * output_array, int methodFlag,
		int size11, int size12, int size13,
		int size21, int size22, int size23,
		int size31, int size32,
		int size41, int size42)

	cdef extern void c_refine(double * ref_hyps,
		double * sampling3D, 
		double * sampling2D, 
		double * objPts, 
		double * imgPts, 
		double * hyps, 
		int * inlier_map,
		int * objIdx, 
		int * shuffleIdx, 
		double * camMat,
		int n, 
		int hyp_num, 
		int init_num, 
		int refSteps, 
		int inlier_count)

	cdef extern void c_refine_single(double * ref_hyp,
		double * sampling3D,
		double * sampling2D,
		double * hyp,
		int * shuffleIdx,
		double * camMat,
		int n,
		int hyp_num,
		int ref_steps,
		int inlier_count)

	cdef extern void c_dRefine(double * jacobean_obj,
		double * jacobean_sample, 
		double * sampling3D, 
		double * sampling2D, 
		double * objPts, 
		double * imgPts,
		int * inlier_map,
		int * objIdx, 
		int * shuffleIdx, 
		double * camMat,
		double * grad,
		int n,
		int hyp_num, 
		int init_num, 
		int ref_steps, 
		int inlier_count,
		double eps,
		int skip)

def solvePnP(np.ndarray[double, ndim=3, mode="c"] input1 not None,
	np.ndarray[double, ndim=3, mode="c"] input2 not None, 
	np.ndarray[double, ndim=2, mode="c"] input3 not None,
	np.ndarray[double, ndim=2, mode="c"] out not None,
	int value):

	cdef int m1, n1, l1, m2, n2, l2, m3, n3, m4, n4

	m1 = input1.shape[0]
	n1 = input1.shape[1]
	l1 = input1.shape[2]

	m2 = input2.shape[0]
	n2 = input2.shape[1]
	l2 = input2.shape[2]

	m3 = input3.shape[0]
	n3 = input3.shape[1]

	m4 = out.shape[0]
	n4 = out.shape[1]


	c_mySolvePnP(&input1[0,0,0], &input2[0,0,0], &input3[0,0], &out[0,0], value, m1, n1, l1, m2, n2, l2, m3, n3, m4, n4)

	return None

def refine(np.ndarray[double, ndim=2, mode="c"] ref_hyps not None,
	    np.ndarray[double, ndim=2, mode="c"] sampling3D not None,
	    np.ndarray[double, ndim=2, mode="c"] sampling2D not None,
	    np.ndarray[double, ndim=3, mode="c"] objPts not None,
	    np.ndarray[double, ndim=3, mode="c"] imgPts not None,
	    np.ndarray[double, ndim=2, mode="c"] hyps not None,
	    np.ndarray[int, ndim=2, mode="c"] inlier_map not None,
	    np.ndarray[int, ndim=2, mode="c"] objIdx not None,
	    np.ndarray[int, ndim=2, mode="c"] shuffleIdx not None,
	    np.ndarray[double, ndim=2, mode="c"] camMat not None,
	    int inlier_count):
	cdef int n, hyp_num, init_num, ref_steps
	n = sampling3D.shape[0]
	hyp_num = objPts.shape[0]
	init_num = objPts.shape[1]
	ref_steps = shuffleIdx.shape[0]
	
	c_refine(&ref_hyps[0,0],
		&sampling3D[0,0],
		&sampling2D[0,0],
		&objPts[0,0,0],
		&imgPts[0,0,0],	
		&hyps[0,0],
		&inlier_map[0,0],
		&objIdx[0,0],
		&shuffleIdx[0,0],
		&camMat[0,0],
		n,
		hyp_num,
		init_num,
		ref_steps,
		inlier_count)

	return None

def refine_single(np.ndarray[double, ndim=1, mode="c"] ref_hyp not None,
	    np.ndarray[double, ndim=2, mode="c"] sampling3D not None,
	    np.ndarray[double, ndim=2, mode="c"] sampling2D not None,
	    np.ndarray[double, ndim=1, mode="c"] hyp not None,
	    np.ndarray[int, ndim=2, mode="c"] shuffleIdx not None,
	    np.ndarray[double, ndim=2, mode="c"] camMat not None,
	    int inlier_count):
	cdef int n, hyp_num, ref_steps
	n = sampling3D.shape[0]
	hyp_num = hyp.shape[0]
	ref_steps = shuffleIdx.shape[0]

	c_refine_single(&ref_hyp[0],
		&sampling3D[0,0],
		&sampling2D[0,0],	
		&hyp[0],
		&shuffleIdx[0,0],
		&camMat[0,0],
		n,
		hyp_num,
		ref_steps,
		inlier_count)

	return None


def dRefine(np.ndarray[double, ndim=3, mode="c"] jacobean_obj not None,
		np.ndarray[double, ndim=2, mode="c"] jacobean_sample not None,
		np.ndarray[double, ndim=2, mode="c"] sampling3D not None,
	    np.ndarray[double, ndim=2, mode="c"] sampling2D not None,
	    np.ndarray[double, ndim=3, mode="c"] objPts not None,
	    np.ndarray[double, ndim=3, mode="c"] imgPts not None,
	    np.ndarray[int, ndim=2, mode="c"] inlier_map not None,
	    np.ndarray[int, ndim=2, mode="c"] objIdx not None,
	    np.ndarray[int, ndim=2, mode="c"] shuffleIdx not None,
	    np.ndarray[double, ndim=2, mode="c"] camMat not None,
	    np.ndarray[double, ndim=2, mode="c"] grad not None,
	    int inlier_count,
	    double eps,
	    int skip):
	
	cdef int n, hyp_num, init_num, ref_steps
	n = sampling3D.shape[0]
	hyp_num = objPts.shape[0]
	init_num = objPts.shape[1]
	ref_steps = shuffleIdx.shape[0]

	c_dRefine(&jacobean_obj[0,0,0],
		&jacobean_sample[0,0],
		&sampling3D[0,0],
		&sampling2D[0,0],
		&objPts[0,0,0],
		&imgPts[0,0,0],	
		&inlier_map[0,0],
		&objIdx[0,0],
		&shuffleIdx[0,0],
		&camMat[0,0],
		&grad[0,0],
		n,
		hyp_num,
		init_num,
		ref_steps,
		inlier_count,
		eps,
		skip)