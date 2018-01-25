from __future__ import division
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import cv2
import glob
import uuid
import math
import time
import copy
import NumericLayers

np.set_printoptions(threshold=np.inf)

CNN_OBJ_MAXINPUT = 100.0
EPS = 0.00000001
REFTIMES = 8
INLIERTHRESHOLD2D = 10
INLIERCOUNT = 20
SKIP = 10
HYPNUM = 1600

def containNan(obj):
	for i in obj:
		if math.isnan(i):
			return True
	return False

def stochasticSubSample(inputImg, targetSize, patchSize):
	(rows, cols) = inputImg.shape
	sampling = np.zeros((targetSize, targetSize, 2))
	xStride = (cols - patchSize) / targetSize
	yStride = (rows - patchSize) / targetSize
	sampleX = 0
	minX = patchSize / 2
	x = xStride + patchSize / 2
	while x <= cols - patchSize / 2 + 1:
		sampleY = 0
		y = yStride + patchSize / 2
		minY = patchSize / 2
		while y <= rows - patchSize / 2 + 1:
			curX = np.random.randint(minX, x)
			curY = np.random.randint(minY, y)
			sampling[sampleY][sampleX][0] = curX
			sampling[sampleY][sampleX][1] = curY
			sampleY += 1
			minY = y
			y += yStride
		sampleX += 1
		minX = x
		x += xStride
	return sampling.reshape(-1, 2)


def cv2our(cvTrans):
	rots = cvTrans[0:3]
	trans = cvTrans[3:6]
	rmat, _ = cv2.Rodrigues(rots)
	#print('----------------------------',trans)
	tpt = copy.deepcopy(trans)
	rmat[1][:] = -rmat[1][:]
	rmat[2][:] = -rmat[2][:]
	tpt[1] = -tpt[1]
	tpt[2] = -tpt[2]
	'''
	if cv2.determinant(rmat) < 0:
		tpt = -tpt
		rmat = -rmat
	'''
	'''
	for i in tpt:
		if math.isnan(i):
			tpt = np.zeros((3,))
	'''
	tpt = tpt.reshape(3,1)
	return np.append(rmat, tpt, axis = 1)


def getRodVecAndTrans(rots, trans):
	rv, _ = cv2.Rodrigues(rots)
	res = np.zeros((6,), np.float32)
	res[0] = rv[0]
	res[1] = rv[1]
	res[2] = rv[2]
	res[3] = trans[0]
	res[4] = trans[1]
	res[5] = trans[2]
	return res

def getRots(hyp):
	return hyp[0:3, 0:3]

def getTrans(hyp):
	return hyp[0:3, 3]

# @param hyp, cvForms
def getDiffMap(hyp, sampling3D, sampling2D, cmat, distcoeffs):
	points2D = sampling2D
	points3D = sampling3D
	projections, _ = cv2.projectPoints(sampling3D, hyp[0:3], hyp[3:6], cmat, distcoeffs)
	(m, _, n) = projections.shape
	projections = projections.reshape(m, n)
	diffPt = points2D - projections
	diffMap = np.minimum(np.linalg.norm(diffPt, axis = 1, keepdims = False), CNN_OBJ_MAXINPUT)
	return diffMap

# Get4Point Layer ---------------------------------------------------------------------------------

def get4point(inputs):
	def _get4point(sampling3D, sampling2D, cmat, distcoeffs):
		objPts = np.zeros((HYPNUM, 4, 3), np.float64)
		imgPts = np.zeros((HYPNUM, 4, 2), np.float64)
		objIdx = np.zeros((HYPNUM, 4), np.int32)
		
		for h in range(HYPNUM):
			#print(h)
			alreadyChosen = np.zeros((sampling3D.shape[0]), np.int)
			while True:
				propObj = np.zeros((4, 3))
				propImg = np.zeros((4, 2))
				propIdx = np.zeros((4,))
				for i in range(4):
					idx = np.random.randint(sampling3D.shape[0])
					#print(idx)
					if alreadyChosen[idx] == 1:
						continue
					propObj[i] = sampling3D[i]
					propImg[i] = sampling2D[i]
					propIdx[i] = idx
					alreadyChosen[idx] = 1

				done, rot, tran = cv2.solvePnP(propObj, propImg, cmat, distcoeffs)
				hyp = np.append(rot, tran)

				projections, _ = cv2.projectPoints(propObj, hyp[0:3], hyp[3:6], cmat, distcoeffs)
				(m, _, n) = projections.shape
				projections = projections.reshape(m, n)
				found = True
				for i in range(4):
					if np.linalg.norm(projections[i] - propImg[i]) < INLIERTHRESHOLD2D:
						continue
					found = False
					break
				if found:
					objPts[h] = propObj
					imgPts[h] = propImg
					objIdx[h] = propIdx
					break
		return objPts, imgPts, objIdx

	def _get4point_grad(sampling3D, objIdx, grad):
		dSample = np.zeros_like(sampling3D)
		for h in range(objIdx.shape[0]):
			dSample[objIdx[h]] = grad[h]
		return dSample

	def _get4point_grad_op(op, grad1, grad2, grad3):
		objIdx = op.outputs[2]
		sampling3D = op.inputs[0]
		tf_dSample = tf.py_func(_get4point_grad, [sampling3D, objIdx, grad1], tf.float64)
		return [tf_dSample, None, None, None]

	grad_name = "Get4pointGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_get4point_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output1, output2, output3 = tf.py_func(_get4point, inputs, [tf.float64, tf.float64, tf.int32])
	return output1, output2, output3


# PnP layer ---------------------------------------------------------------------------------------
# @param objPts, shape [h, 4, 3]
def pnp(inputs):
	def _pnp(objPts, imgPts, cmat, distcoeffs):
		global fff, e2
		if fff:
			e2 = time.time()
			fff = False
		res = np.zeros((objPts.shape[0], 6),np.float64)
		NumericLayers.solvePnP(objPts, imgPts, cmat, res, 2)
		return res
	
	def _pnp_grad(objPts, imgPts, cmat, distcoeffs, grad):
		(h_len, m, n) = objPts.shape
		jacobean = np.zeros((h_len, m, n), np.float64)
		eps = 1
		obj_0 = np.tile(objPts, (12, 1, 1))
		obj_1 = copy.deepcopy(obj_0)

		for i in range(4):
			for j in range(3):
				obj_0[(i*3+j)*h_len:(i*3+j+1)*h_len, i, j] -= eps
				obj_1[(i*3+j)*h_len:(i*3+j+1)*h_len, i, j] += eps
		
		tmp_obj = np.concatenate((obj_0, obj_1))
		tmp_img = np.tile(imgPts, (24, 1, 1))
		tmp_hyp = np.zeros((tmp_obj.shape[0], 6), np.float64)

		NumericLayers.solvePnP(tmp_obj, tmp_img, cmat, tmp_hyp, 2)
		
		res_0 = tmp_hyp[0:h_len*12, :]
		res_1 = tmp_hyp[h_len*12:h_len*24,:]

		for i in range(4):
			for j in range(3):
				jacobean[:, i, j] = np.sum(grad * (res_1[(i*3+j)*h_len:(i*3+j+1)*h_len,:] - res_0[(i*3+j)*h_len:(i*3+j+1)*h_len,:]) / (2 * eps), axis = 1)
		
		return jacobean

	def _pnp_grad_op(op, grad):
		aa = op.inputs[0]
		bb = op.inputs[1]
		cc = op.inputs[2]
		dd = op.inputs[3]
		p_grad = tf.py_func(_pnp_grad, [aa, bb, cc, dd, grad], tf.float64)
		return [p_grad, None, None, None]

	grad_name = "PnPGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_pnp_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output = tf.py_func(_pnp, inputs, tf.float64)
	return output

# reprojection error layer --------------------------------------------------------------------
def reprojection(inputs):
	# @param objPts, 3D coordinates
	# @param hyps, cvForms, 1x6, containing rvec and tvec
	# @param imgPts, 2D coordinates
	def _reprojection(sampling3D, hyps, sampling2D, cmat, distcoeffs):
		#print(objPts)
		(h, _) = hyps.shape
		(n, _) = imgPts.shape
		diffMaps = np.zeros((h, n))
		for i in range(h):
			diffMaps[i] = getDiffMap(hyps[i], sampling3D, sampling2D, cmat, distcoeffs)
		return diffMaps

	# @param hyp, jpForms, 3x3 + 3x1, rmat and tvec
	def _dProject_wrt_obj(sampling3D, hyp, sampling2D, cmat, grad):
		# camera instrincs
		f = cmat[0][0]
		ppx = cmat[0][2]
		ppy = cmat[1][2]
		rot = getRots(hyp)
		tran = getTrans(hyp)
		jacobean = np.zeros_like(sampling3D)

		for i in range(sampling3D.shape[0]):
			objMat = np.dot(rot, sampling3D[i]) + tran

			if np.abs(objMat[2]) < EPS:
				continue
			px = -f * objMat[0] / objMat[2] + ppx
			py = f * objMat[1] / objMat[2] + ppy

			err = np.sqrt((sampling2D[i][0] - px) ** 2 + (sampling2D[i][1] - py) ** 2)
			if err > CNN_OBJ_MAXINPUT:
				continue

			err += EPS

			pxdx = -f * rot[0][0] / objMat[2] + f * objMat[0] / objMat[2] / objMat[2] * rot[2][0]
			pydx = f * rot[1][0] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][0]
			dx = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdx) + 2 * (sampling2D[i][1] - py) * (-pydx))

			pxdy = -f * rot[0][1] / objMat[2] + f *objMat[0] / objMat[2]/ objMat[2] * rot[2][1]
			pydy = f * rot[1][1] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][1]
			dy = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdy) + 2 * (sampling2D[i][1] - py) * (-pydy))

			pxdz = -f * rot[0][2] / objMat[2] + f *objMat[0] / objMat[2]/ objMat[2] * rot[2][2]
			pydz = f * rot[1][2] / objMat[2] - f * objMat[1] / objMat[2] / objMat[2] * rot[2][2]
			dz = 0.5 / err * (2 * (sampling2D[i][0] - px) * (-pxdz) + 2 * (sampling2D[i][1] - py) * (-pydz))

			jacobean[i][0] = dx * grad[i]
			jacobean[i][1] = dy * grad[i]
			jacobean[i][2] = dz * grad[i]

		return jacobean

	# @param hyp, jpForms, 3x3+3x1, rmat and tvec
	def _dProject_wrt_hyp(sampling3D, hyp, sampling2D, cmat, grad):
		f = cmat[0][0]
		ppx = cmat[0][2]
		ppy = cmat[1][2]
		rot = getRots(hyp)
		tran = getTrans(hyp)
		jacobean = np.zeros((6,))
		dRdH = np.zeros((9, 3))
		rot1 = copy.deepcopy(rot)
		rot1[1][:] = -rot1[1][:]
		rot1[2][:] = -rot1[2][:]
		rod, _ = cv2.Rodrigues(rot1)
		rot1, dRdH = cv2.Rodrigues(rod)
		dRdH = dRdH.T

		for i in range(sampling3D.shape[0]):
			objMat = sampling3D[i]
			eyeMat = np.dot(rot, sampling3D[i]) + tran
			if np.abs(eyeMat[2]) < EPS:
				print("fire")
				continue
			
			px = -f * eyeMat[0] / eyeMat[2] + ppx
			py = f * eyeMat[1]/ eyeMat[2] + ppy

			err = np.sqrt((sampling2D[i][0] - px)**2 + (sampling2D[i][1] - py)**2)

			if err > CNN_OBJ_MAXINPUT:
				continue

			err += EPS

			# derivative of error wrt projection
			dNdP = np.zeros((2,))
			dNdP[0] = -1 / err * (sampling2D[i][0] - px) * grad[i]
			dNdP[1] = -1 / err * (sampling2D[i][1] - py) * grad[i]
			# derivative of projection wrt rotation matrix
			dPdR = np.zeros((2, 9))
			
			dPdR[0][0:3] = -f * objMat / eyeMat[2]
			dPdR[1][3:6] = f * objMat / eyeMat[2]
			dPdR[0][6:9] = f * eyeMat[0] / eyeMat[2] / eyeMat[2] * objMat
			dPdR[1][6:9] = -f * eyeMat[1] / eyeMat[2]/ eyeMat[2] * objMat
			dPdR[:,3:9] = -dPdR[:, 3:9]
			
			# derivative of rotation matrix wet rodriguez vector
			# combine, derivative of error wrt Rodriguez vector
			dNdH = np.dot(np.dot(dNdP, dPdR), dRdH)

			# derivative of projection wet translation vector
			dPdT = np.zeros((2,3))
			dPdT[0][0] = -f / eyeMat[2]
			dPdT[1][1] = f / eyeMat[2]
			dPdT[0][2] = f * eyeMat[0] / eyeMat[2] / eyeMat[2]
			dPdT[1][2] = -f * eyeMat[1] / eyeMat[2] / eyeMat[2]

			dNdT = np.dot(dNdP, dPdT)
			dNdT[1:3] = -dNdT[1:3]
			jacobean[0:3] += dNdH
			jacobean[3:6] += dNdT

		return jacobean

	def _reprojection_grad(sampling3D, hyps, sampling2D, cmat, distcoeffs, grad):
		dObj = np.zeros_like(sampling3D)
		dHyp = np.zeros_like(hyps)
		for i in range(hyps.shape[0]):
			ourHyp = cv2our(hyps[i])
			dObj += _dProject_wrt_obj(sampling3D, ourHyp, sampling2D, cmat, grad[i])
		for i in range(hyps.shape[0]):
			ourHyp = cv2our(hyps[i])
			dHyp[i] = _dProject_wrt_hyp(sampling3D, ourHyp, sampling2D, cmat, grad[i])
		return dObj, dHyp

	def _reprojection_grad_op(op, grad):
		sampling3D = op.inputs[0]
		hyps = op.inputs[1]
		sampling2D = op.inputs[2]
		cmat = op.inputs[3]
		distcoeffs = op.inputs[4]
		objGrad, hypGrad = tf.py_func(_reprojection_grad, [sampling3D, hyps, sampling2D, cmat, distcoeffs, grad], [tf.float64, tf.float64])
		return [objGrad, hypGrad, None, None, None]
	grad_name = "reprojectionGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_reprojection_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output = tf.py_func(_reprojection, inputs, tf.float64)
	return output

# refine layer ---------------------------------------------------------------------------------
def refine(inputs):
	in_obj = inputs[0]
	in_hyp = inputs[4]
	inlierMaps = np.zeros((in_hyp.shape[0], in_obj.shape[0]))
	def _refine(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
		refHyps = np.zeros(hyps.shape, dtype=np.float64)	
		samplingCopy = copy.deepcopy(sampling3D)
		
		#for i in range(objPts.shape[0]):
		#	samplingCopy[objIdx[i]] = copy.deepcopy(objPts[i])
		
		e1 = time.time()
		for h in range(refHyps.shape[0]):
			done0, rot0, tran0 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)
			newHyp = np.append(rot0, tran0)
			diffmaps = getDiffMap(newHyp, samplingCopy, sampling2D,cmat, distcoeffs)
			for i in range(REFTIMES):
				inlier3D = []
				inlier2D = []
				for idx in shuffleIdx[i]:
					if diffmaps[idx] < INLIERTHRESHOLD2D:
						inlier3D.append(samplingCopy[idx])
						inlier2D.append(sampling2D[idx])
						inlierMaps[h][idx] = 1
					if len(inlier3D) >= INLIERCOUNT:
						break
				if len(inlier3D) < 3:
					continue
				refineObj = np.array(inlier3D)
				refinePt = np.array(inlier2D)
				done, rot, tran = cv2.solvePnP(refineObj, refinePt, cmat, distcoeffs, False, cv2.SOLVEPNP_ITERATIVE if refineObj.shape[0] >= 4 else cv2.SOLVEPNP_P3P)
				
				if containNan(rot) or containNan(tran):
					break
				refHyps[h] = np.append(rot, tran)
				diffmaps = getDiffMap(refHyps[h], samplingCopy, sampling2D, cmat, distcoeffs)
			
			for idx in objIdx[h]:
				inlierMaps[h][idx] = 0
		
		print "python use", time.time() - e1

		e2 = time.time()
		refHyps1 = np.zeros(hyps.shape, dtype=np.float64)
		NumericLayers.refine(refHyps1,
			sampling3D,
			sampling2D,
			objPts,
			imgPts,
			hyps,
			objIdx,
			shuffleIdx,
			cmat,
			INLIERCOUNT)
		print "cython use", time.time() - e2

		print "err:", refHyps - refHyps1
		return refHyps

	def _refine_single(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
		refHyps = np.zeros(hyps.shape, dtype=np.float64)
		for h in range(refHyps.shape[0]):
			diffmaps = getDiffMap(hyps[h], sampling3D, sampling2D, cmat, distcoeffs)
			for i in range(REFTIMES):
				inlier3D = []
				inlier2D = []
				for idx in shuffleIdx[i]:
					if diffmaps[idx] < INLIERTHRESHOLD2D:
						inlier3D.append(sampling3D[idx])
						inlier2D.append(sampling2D[idx])
					if len(inlier3D) >= INLIERCOUNT:
						break
				if len(inlier3D) < 3:
					continue
				refineObj = np.array(inlier3D)
				refinePt = np.array(inlier2D)
				done, rot, tran = cv2.solvePnP(refineObj, refinePt, cmat, distcoeffs, False, cv2.SOLVEPNP_ITERATIVE if refineObj.shape[0] >= 4 else cv2.SOLVEPNP_P3P)
				if containNan(rot) or containNan(tran):
					break
				refHyps[h] = np.append(rot, tran)
				#print("i:", i, refHyps[h])
				diffmaps = getDiffMap(refHyps[h], sampling3D, sampling2D, cmat, distcoeffs)
		return refHyps

	# @param grad, dLoss wrt hyp, hx6, actually the same shape as hyps
	def _refine_grad(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs, grad):
		#print(grad)
		# dRefine wrt the picked up points, which were used to generate pose before
		# numeric method
		eps = 1
		jacobean_obj = np.zeros((objPts.shape[0], objPts.shape[1], objPts.shape[2]), np.float64)
		jacobean_sample = np.zeros_like(sampling3D, np.float64)
		e1 = time.time()
		'''
		for h in range(in_hyp.shape[0]):
			for i in range(objPts.shape[1]):
				for j in range(3):
					# forward step
					objPts[h][i][j] += eps
					sampling3D[objIdx[h][i]][j] += eps
					done, rot1, tran1 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)

					newHyp = np.append(rot1, tran1)

					fstep = np.zeros((6,),dtype=np.float64)
					fstep = _refine_single(sampling3D, sampling2D, objPts, imgPts, np.array([newHyp]), objIdx, shuffleIdx, cmat, distcoeffs)
					
					# backward step
					objPts[h][i][j] -= 2 * eps
					sampling3D[objIdx[h][i]][j] -= 2 * eps

					done, rot2, tran2 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)
					newHyp2 = np.append(rot2, tran2)

					bstep = np.zeros((6,), dtype=np.float64)

					bstep = _refine_single(sampling3D, sampling2D, objPts, imgPts, np.array([newHyp2]), objIdx, shuffleIdx, cmat, distcoeffs)
					objPts[h][i][j] += eps
					sampling3D[objIdx[h][i]][j] += eps

					dLocal = (fstep - bstep) / (2 * eps)
					jacobean_obj[h][i][j] = np.sum(grad[h] * dLocal)
		
                print "python use", time.time() - e1
		'''

		jacobean_obj1 = np.zeros_like(objPts)
		e2 = time.time()
		NumericLayers.dRefine(jacobean_obj1,
			sampling3D,
			sampling2D,
			objPts,
			imgPts,
			objIdx,
			shuffleIdx,
			cmat,
			grad,
			INLIERCOUNT,
			eps)
		print "cython use", time.time() - e2

		# dRefine wrt other points
		'''
		for h in range(in_hyp.shape[0]):

			inCount = 0
			for i in range(sampling3D.shape[0]):
				
				if inlierMaps[h][i] == 0:
					continue
				
				inCount += 1
				if inCount % SKIP == 0:
					continue
				for j in range(3):
					# forward step
					sampling3D[i][j] += eps
					done, rot1, tran1 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)
					newHyp = np.append(rot1, tran1)	
					fstep = _refine_single(sampling3D, sampling2D, objPts, imgPts, np.array([newHyp]), objIdx, shuffleIdx, cmat, distcoeffs)
					# backward step
					sampling3D[i][j] -= 2 * eps
					done, rot2, tran2 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)
					newHyp2 = np.append(rot2, tran2)
					bstep = _refine_single(sampling3D, sampling2D, objPts, imgPts, np.array([newHyp2]), objIdx, shuffleIdx, cmat, distcoeffs)

					sampling3D[i][j] += eps

					dLocal = (fstep - bstep) / (2 * eps)
					jacobean_sample[i][j] += np.sum(grad[h] * dLocal)
		'''
		return jacobean_obj, jacobean_sample

	def _refine_grad_op(op, grad):
		[sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs] = op.inputs

		dObj, dSample = tf.py_func(_refine_grad, [sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs, grad], [tf.float64, tf.float64])		
		return [dSample, None, dObj, None, None, None, None, None, None]
	
	grad_name = "RefineGrad_" + str(uuid.uuid4())
	tf.RegisterGradient(grad_name)(_refine_grad_op)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": grad_name}):
		output = tf.py_func(_refine, inputs, tf.float64)
	return output


# Testing ! for refine layer~~

xyz = [[10, 22, 3], [12, 5, 9], [20, 20, 14], [4, 12, 23]]
xy = [[0, 1], [2, 2], [3, 3], [4, 4]]
# hyps = [[20, 10, 20, 0.1, 0.5, 1.2], [10, 2, 8, 3.2, 4.2, 2.3]]
D = np.array([[0, 0, 0, 0, 0]], np.float64)
camera_matrx = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
np_xyz = np.array(xyz, np.float64)
np_cmat = np.array(camera_matrx, np.float64)
np_xy = np.array(xy, np.float64)
#np_hyp = np.array(hyps, np.float64)
tf_xyz = tf.constant(np_xyz)
tf_xy = tf.constant(np_xy)
tf_cmat = tf.constant(np_cmat)
tf_D = tf.constant(D)
#tf_hyp = tf.constant(np_hyp)

# h = 4
# sampling num N = 100
hyp_num = HYPNUM
sample = np.random.uniform(1, 10, size=(5, 3))
sampling3D1 = np.random.uniform(0, 20, size=(100, 3))
sampling2D1 = np.random.uniform(0, 5, size=(100, 2))
objIdx1 = np.random.randint(100, size=(hyp_num, 4))
objIdx1 = np.cast['int32'](objIdx1)
shuffleIdx1 = np.zeros((8, 100), dtype=np.int)
for i in range(shuffleIdx1.shape[0]):
	shuffleIdx1[i] = np.arange(100)
	np.random.shuffle(shuffleIdx1[i])
shuffleIdx1 = np.cast['int32'](shuffleIdx1)
objPts1 = copy.deepcopy(sampling3D1[objIdx1])
imgPts1 = copy.deepcopy(sampling2D1[objIdx1])
in_hyps = np.zeros((hyp_num, 6))
for h in range(hyp_num):
	done0, rot0, tran0 = cv2.solvePnP(objPts1[h], imgPts1[h], np_cmat, distCoeffs=None)
	in_hyps[h] = np.append(rot0, tran0)
diffMaps1 = np.zeros((hyp_num, 100))
#for i in range(1):
#	diffMaps1[i] = getDiffMap(in_hyps[i], sampling3D1, sampling2D1, np_cmat, D)

tf_objIdx = tf.constant(objIdx1)
tf_diffMaps = tf.constant(diffMaps1)
tf_sample3D = tf.constant(sampling3D1)
tf_sample2D = tf.constant(sampling2D1)
tf_hyp = tf.constant(in_hyps)
tf_objPts = tf.constant(objPts1)
tf_imgPts = tf.constant(imgPts1)
tf_shuffleIdx = tf.constant(shuffleIdx1)
out = refine([tf_sample3D, tf_sample2D, tf_objPts, tf_imgPts, tf_hyp, tf_objIdx, tf_shuffleIdx, tf_cmat, tf_D])
config = tf.ConfigProto(device_count={'GPU':0})
with tf.Session(config=config) as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	fff = True
	e1 = time.time()
	e2 = 0
	diff = tf.test.compute_gradient_error(tf_objPts, [hyp_num, 4, 3], out, [hyp_num, 6], delta=1, x_init_value=objPts1)
	#a = sess.run(out)
	#print "cython use time per backprop: ", (e2 - e1) / (hyp_num * 6)
	#print "tensorflow use time per backbrop: ", (time.time() - e2) / (hyp_num * 6)
	#print(diff)
	#print("---------------")
	#print(diff[1])

