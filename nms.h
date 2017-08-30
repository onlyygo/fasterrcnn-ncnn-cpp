#ifndef NMS_H_
#define NMS_H_

#include <map>
#include <vector>

template<typename T>
void nms(const T *pBoxes, const T *pScores, int nSample, double overlap,
		int &nPick, T* &boxes_nms, T* &scores_nms) {
	const int BOX_DIMS = 4;
	std::vector<int> vPick(nSample);
	std::vector<double> vArea(nSample);
	for (int i = 0; i < nSample; ++i) {
		vArea[i] = double(
				pBoxes[i * BOX_DIMS + 2] - pBoxes[i * BOX_DIMS + 0] + 1)
				* (pBoxes[i * BOX_DIMS + 3] - pBoxes[i * BOX_DIMS + 1] + 1);
		if (vArea[i] < 0)
			printf("Boxes area must >= 0");
	}
	std::multimap<T, int> scores;
	for (int i = 0; i < nSample; ++i)
		scores.insert(std::pair<T, int>(pScores[i], i));

	nPick = 0;
	do {
		int last = scores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;

		for (typename std::multimap<T, int>::iterator it = scores.begin();
				it != scores.end();) {
			int it_idx = it->second;
			T xx1 = std::max(pBoxes[0 + BOX_DIMS * last],
					pBoxes[0 + BOX_DIMS * it_idx]);
			T yy1 = std::max(pBoxes[1 + BOX_DIMS * last],
					pBoxes[1 + BOX_DIMS * it_idx]);
			T xx2 = std::min(pBoxes[2 + BOX_DIMS * last],
					pBoxes[2 + BOX_DIMS * it_idx]);
			T yy2 = std::min(pBoxes[3 + BOX_DIMS * last],
					pBoxes[3 + BOX_DIMS * it_idx]);

			double w = std::max(T(0.0), xx2 - xx1 + 1), h = std::max(T(0.0),
					yy2 - yy1 + 1);

			double ov = w * h / (vArea[last] + vArea[it_idx] - w * h);

			if (ov > overlap) {
				it = scores.erase(it);
			} else {
				it++;
			}
		}

	} while (scores.size() != 0);
	if (boxes_nms != NULL)
		delete[] boxes_nms;
	boxes_nms = new T[nPick * BOX_DIMS];
	if (scores_nms != NULL)
		delete[] scores_nms;
	scores_nms = new T[nPick];
	for (int i = 0; i < nPick; ++i) {

		int index = vPick[i];
		boxes_nms[i * BOX_DIMS + 0] = pBoxes[index * BOX_DIMS + 0];
		boxes_nms[i * BOX_DIMS + 1] = pBoxes[index * BOX_DIMS + 1];
		boxes_nms[i * BOX_DIMS + 2] = pBoxes[index * BOX_DIMS + 2];
		boxes_nms[i * BOX_DIMS + 3] = pBoxes[index * BOX_DIMS + 3];
		scores_nms[i] = pScores[index];
	}
}
#endif