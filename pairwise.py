import cv2

from matplotlib import pyplot as plt

def ranking_by_color_hist( key_img, data ):

	query_hist = cv2.calcHist([key_img], [0], None, [16], [0, 256])
	query_hist = cv2.normalize(query_hist, query_hist).flatten()

#	plt.figure()
#	plt.title("Grayscale Histogram")
#	plt.xlabel("Bins")
#	plt.ylabel("# of Pixels")
#	plt.plot(cv2.calcHist([key_img], [0], None, [256], [0, 256]))
#	plt.xlim([0, 255])
#	plt.show()
#
#	print("hist: ",hist)

	images_hist = {}
	for elem in data:
		hist = cv2.calcHist([elem['image']], [0], None, [16], [0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		images_hist[elem['imgname']] = hist
	
	results = {}
	for (name, hist) in images_hist.items():
		dist = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
		results[name] = dist


	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()], reverse = True)

	sorted_names = []
	for (i, (v, k)) in enumerate(results):
		print(i,v,k)
		sorted_names.append(k)

	return sorted_names