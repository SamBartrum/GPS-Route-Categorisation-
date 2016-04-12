from __future__ import division
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict


# Each journey is an instance of the Journey class
class Journey(object):

	# Class member list to store all instances i.e. data sets
	JournList = []

	# Set to store the unique journeys i.e. ones recorded once
	unique = set([])

	# Duplicate routes i.e. the same journey recorded with multiple devices
	duplicate = []

	# Acts as our bin for splitting the journeys up
	hashed = defaultdict(lambda: [])

	# Constructor for a single journey
	def __init__(self, journey):

		# Read in the csv file
		data = pd.read_csv(journey, names = ["TIME" , "LAT" , "LON" , "ALT" , "ACC" , "GPST" , "SPEED"])

		# storing for plotting only, if plotting is not needed the whole data set need not be saved
		self.data = data
		self.path = journey

		# Variances for the various quantities
		self.SPEED_var = np.var(data.SPEED)
		self.LAT_var   = np.var(data.LAT)
		self.LON_var   = np.var(data.LON)

		# Derived median properties
		self.LAT_med = np.median(data.LAT)
		index = data.LAT.index[self.LAT_med]

		# Error on the median value
		self.LAT_med_error = Journey.convertToLat(data.ACC[index])

		self.LON_med = np.median(data.LON)
		index = data.LON.index[self.LON_med]
		self.LON_med_error = Journey.convertToLong(data.ACC[index], data.LAT[index])

		# Add the journey to the list
		Journey.JournList.append(self)

		# When we are done constructing we stick the object into a bin
		self.hash_()


	# Convert the uncertainty in meters to an uncertainty in latitude (assume spherical earth for now)
	@staticmethod
	def convertToLat(distance):

		# Earth's mean radius in meters
		r_E = 6371000.0

		return distance * (4 * 90 / (2 * math.pi)) / r_E

	# Convert uncertainty in meters into uncertainty in longitude, we need the latitude for this (again assume spherical
	# earth) blows up at the north pole (not an issue here but should be dealt with in a more complete algorithm)
	@staticmethod
	def convertToLong(distance, latitude):

		# Earth's mean radius in meters
		r_E = 6371000.0

		# Convert to radians
		lat_rad = latitude * 2 * math.pi / 360.0

		# The circumerence of a circle at a given latitude
		circle_r = 2 * math.pi * r_E * math.cos(lat_rad)

		# Return the error in degrees
		return distance * 360.0 / circle_r



	# A simple hashing function to dump journeys with similar median Lat/Lon coordinates into the same bin
	# this reduces the time complexity of the comparisons significantly as we only compare within each bin.
	# The only worry is that two of the same journey may be split over the two bins due to the uncertainty
	# in the measurements - but this can be avoided by placing the route in all possible bins and removing duplicates.
	# The problem lies in choosing a hashing strategy which is sufficiently subtle but not too subtle.  Rounding to the nearest
	# two decimal places equates to the nearest ~km and this seems sufficient for this task.
	def hash_(self):

		# Round the median longitude/latitude pairs to 2 decimal places
		x = round(self.LON_med, 2)
		y = round(self.LAT_med, 2)

		# We stick it in the dictionary/bin
		Journey.hashed[(x,y)].append(self)

		# We catch the cases which may be split between bins by hashing them into all possible
		# bins - we will make sure not to over count these routes twice later i.e. use sets
		xupper = round(self.LON_med + self.LON_med_error, 2)
		xlower = round(self.LON_med - self.LON_med_error, 2)

		yupper = round(self.LAT_med + self.LAT_med_error, 2)
		ylower = round(self.LAT_med - self.LAT_med_error, 2)

		# If the route could be in another bin within its error - then we put it in that bin also
		if x != xupper:
			Journey.hashed[(xupper, y)].append(self)
		if x != xlower:
			Journey.hashed[(xlower, y)].append(self)

		if y != yupper:
			Journey.hashed[(x, yupper)].append(self)
		if y != ylower:
			Journey.hashed[(x, ylower)].append(self)


	# Reads in the journeys from the data directory
	@classmethod
	def readJourneys(cls, directory, extension):

		# List of file locations
		journeys = [directory + f for f in listdir(directory) if extension in f]

		for journey in journeys:
			cls(journey)


	# Returns the unique routes
	@classmethod
	def uniqueRoutes(cls):

		for x in cls.JournList:
			contained = False

			for y in cls.duplicate:
				if x in y:
					contained = True

			if contained == False:
				cls.unique.add(x)

	# Now we sift through the bins which contain many fewer journeys
	@classmethod
	def sift(cls):

		# Cycle through the bins
		for key in cls.hashed:
			sublist = cls.hashed[key]

			# If there is only one item in the bin it is a unique route/journey so we pass
			if len(sublist) == 1: continue

			# Otherwise we compare further
			else:
				for i in range(len(sublist) - 1):

					a = sublist[i]

					temp = set([])

					for j in range(i + 1, len(sublist)):

						b = sublist[j]

						# This works pretty well to pair up the duplicate routes.
						# Basically ensuring that the data variances are within 10% of each other and 1% for the speed
						if abs((a.LAT_var / b.LAT_var) - 1) < 0.1 and \
						   abs((a.LON_var / b.LON_var) - 1) < 0.1 and \
						   abs((a.SPEED_var / b.SPEED_var) - 1) < 0.01:

							temp.update([a, b])

					# The prevents the addition of an empty set as well as duplicate sets in the not
					# too common event where two pairs of journeys get hashed into multiple bins
					if temp and temp not in cls.duplicate:
						cls.duplicate.append(temp)

		# There are scenarios with duplicate routes split between bins, we need to combine these sets together,
		# for example Journey.duplicate = [ set(1,2), set(2,3) ] and we want [ set(1,2,3) ]
		A = cls.duplicate
		B = cls.duplicate

		# True if there are any overlapping sets in a list
		merges = any(a & b for a in A for b in A if a != b)

		# While we can still merge the list of sets, do so
		while merges:
			B = [A[0]]
			for a in A[1:]:
				merged = False
				for i, b in enumerate(B):

					# If the sets overlap
					if a & b:
						B[i] = b | a
						merged = True
						break

				# If the sets do not overlap
				if not merged:
					B.append(a)

			A = B

			# Check for more merges
			merges = any(a & b for a in A for b in A if a != b)

		# Assign the duplicates to the result of this while loop
		cls.duplicate = B




# Read in the data and hash them into bins
Journey.readJourneys('data/','.txt')

# Sift the journeys
Journey.sift()

# Find the unique routes
Journey.uniqueRoutes()

print "Number of duplicates sets:   " + str(len(Journey.duplicate))
print "Number of duplicated routes: " + str(sum([len(x) for x in Journey.duplicate]))
print "Number of unique journeys:   " + str(len(Journey.unique))

# Plot the duplicate journeys
for x in Journey.duplicate:
	for y in x:
		plt.plot(y.data.LAT, y.data.LON)

plt.show()

# Plot the unique journeys
for x in Journey.unique:
	plt.plot(x.data.LAT, x.data.LON)

plt.show()
