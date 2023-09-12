# Gotta-Go-Fast
Caffeine runs through your veins and you can’t function without it. No matter where you go, you would need to grab a cup of coffee on the way to the destination. You, however, often find yourself late to your destination due to this; especially when the cafes have a long waiting time. With that, you have downloaded the travel time between key locations in your city which you can use to estimate the travel time from a point to another. You have also time the waiting time of each cafe that you would grab your coffee from. 

Therefore we create a RoadGraph class :
- The init method of RoadGraph would take as an input a list of roads roads represented as a list of tuples (u,v,w) where:
  - u is the starting location ID for a road, represented as a non-negative integer.
  -  v is the ending location ID for a road, represented as a non-negative integer
  -  w is the time taken to travel from location u to location v, represented as a non-negative integer
  -  We didn't assume that the list of tuples are in any specific order
  -  We didn't assume that the roads are 2-way roads

We can then calculate the optimal routes for your commute while grabbing coffee along the way using a function in the RoadGraph class called routing(self, start, end). The init method of RoadGraph also takes as an input a list of cafes cafes represented as a list of tuples (location,waiting_time) where:
  - Location is the location of the cafe; represented as a non-negative integer
  - waiting_time is the waiting time for a coffee in the cafe, represented as a non-negative integer
  - We didn't assume that the list of tuples are in any specific order.
  - We assumed that all of the location values are from the set {0, 1, ..., |V|-1}
  - We assumed that all of the location values are unique
  - We didn't assume waiting_time to be within any range except that it is a value > 0

# Maximum Score
You are an avid snowboarder and are doing your best efforts to prepare for the upcoming local
tournament.
You made several visits to the resort where the tournament will take place and cautiously studied the trails and practised on them in oder to determine the score that you can obtain in each segment. With your current skill level you are able to move downhill, but you are not able to move properly neither uphill nor on flat surfaces. Therefore, during the tournament you will only go through downhill segments (i.e., the start point of the segment is higher than the end point).
Given your extensive preparation efforts, we are able to determine with perfect precision the score you would be able to get on each downhill segment if we decide to go through it. Now, given the start and end points of the tournament, we want to maximise our score by choosing the best combination of downhill segments to go from the start point to the end point so that the sum of your scores in the used segments is maximised.
Each segment starts and finishes at an intersection point. There are |P| intersection points and they are denoted by 0, 1, . . . , |P| − 1. There are |D| downhill segments. You can assume that for each intersection point there is at least one downhill segment that starts or finishes at that intersection point.
From your preparation, we learned downhillScores, which is represented as a list of |D| tuples (a, b, c):
  - a is the start point of a downhill segment, a ∈ {0, 1, . . . , |P| − 1}
  - b is the end point of a downhill segment, b ∈ {0, 1, . . . , |P| − 1}
  - c is the integer score that you would get for using this downhill segment to go from point a to point b
  - We cannot assume that the list of tuples are in any specific order.

Now, the tournament organisers have decided the starting point start and finishing point finish of the tournament. We have that start, finish ∈ {0, 1, . . . , |P| − 1}. We implemented a function **optimalRoute(downhillScores, start, finish)** that outputs the route that you should use for going from the starting point start to the finishing point finish while using only downhill segments and obtaining the maximum score:
 - If no such route going from the starting point start to finishing point finish while using only downhill segments exists, then the function would return None.
 - Otherwise, it would return the optimal route as a list of integers. If there are multiple optimal routes, return any of them.
