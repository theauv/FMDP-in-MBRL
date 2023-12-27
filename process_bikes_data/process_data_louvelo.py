import pandas as pd
import math
from tqdm import tqdm

# # trips_centroids_coords = pd.read_csv(
# #     "process_bikes_data/Louisville_Metro_KY_-_Feature_Class_Containing_Trip_Level_Data_for_LouVelo_Bikeshare_Program.csv",
# #     usecols=["Start_Station_Id","Start_Latitude","Start_Longitude","End_Station_Id","End_Latitude","End_Longitude"]
# # )

# # trips_centroids_coords = trips_centroids_coords.reset_index()  # make sure indexes pair with number of rows
# # num_centroids = max(trips_centroids_coords["Start_Station_Id"].max(axis=0), trips_centroids_coords["End_Station_Id"].max(axis=0))+1
# # centroids_coords = np.zeros((num_centroids, 2))
# # centroids_count = np.zeros((num_centroids, 2))

# # for index, row in trips_centroids_coords.iterrows():
# #     #Start_station
# #     start_station_id = row["Start_Station_Id"]
# #     start_latitude = row["Start_Latitude"]
# #     start_longitude = row["Start_Longitude"]
# #     print(start_station_id, start_latitude, start_longitude)
# #     if not math.isnan(start_station_id):
# #         start_station_id = int(start_station_id)
# #         if not math.isnan(start_latitude) and centroids_count[start_station_id, 0]==0.:
# #             centroids_count[start_station_id, 0] = 1.
# #             centroids_coords[start_station_id, 0] = start_latitude
# #         if not math.isnan(start_longitude) and centroids_count[start_station_id, 1]==0.:
# #             centroids_count[start_station_id, 1] = 1.
# #             centroids_coords[start_station_id, 1] = start_longitude
# #     #End_station
# #     end_station_id = row["End_Station_Id"]
# #     end_latitude = row["End_Latitude"]
# #     end_longitude = row["End_Longitude"]
# #     print(end_station_id, end_latitude, end_longitude)
# #     if not math.isnan(end_station_id):
# #         end_station_id = int(end_station_id)
# #         if not math.isnan(end_latitude) and centroids_count[end_station_id, 0]==0.:
# #             centroids_count[end_station_id, 0] = 1.
# #             centroids_coords[end_station_id, 0] = end_latitude
# #         if not math.isnan(end_longitude) and centroids_count[end_station_id, 1]==0.:
# #             centroids_count[end_station_id, 1] = 1.
# #             centroids_coords[end_station_id, 1] = end_longitude

# #     if np.all(centroids_count):
# #         break

# # centroids_coords = centroids_coords[~np.all(centroids_coords==0., axis=1)]
# # np.save("centroids_coords", centroids_coords)

# #Create the "all_trips_data"
# # all_trips_data = pd.read_csv(
# #     "process_bikes_data/Louisville_Metro_KY_-_Feature_Class_Containing_Trip_Level_Data_for_LouVelo_Bikeshare_Program.csv"
# # )

# # all_trips_data = all_trips_data.drop(["X","Y","Closed_Status","Bike_Barcode","Bike_Model","Product_Name","ObjectId"], axis=1)

# # def date_and_time(row):
# #     start_date = str(row["Start_Date"])
# #     start_date, start_time = start_date.split(" ")
# #     start_time = start_time.split(":")[:2]
# #     start_time = ':'.join(start_time)
# #     year, month, day = start_date.split("/")
# #     row['Year'] = year
# #     row['Month'] = month
# #     row['day'] = day
# #     row['StartTime'] = start_time
# #     return row

# # all_trips_data["Year"] = ""
# # all_trips_data["Month"] = ""
# # all_trips_data["Day"] = ""
# # all_trips_data["StartTime"] = ""
# # tqdm.pandas()
# # all_trips_data=all_trips_data.progress_apply(date_and_time, axis=1)
# # #all_trips_data = all_trips_data.apply(date_and_time, axis=1)
# # all_trips_data.to_csv("all_trips_LouVelo_recent.csv", index=False)

# all_trips_data = pd.read_csv(
#     "src/env/bikes_data/all_trips_LouVelo_recent.csv"
# )

# all_trips_data = all_trips_data.rename(columns={"Duration": "TripDuration","Start_Latitude": "StartLatitude","Start_Longitude": "StartLongitude","End_Longitude": "EndLongitude", "End_Latitude": "EndLatitude"})

# print(all_trips_data.columns)

# tqdm.pandas()
# all_trips_data["DayOfWeek"] = all_trips_data.progress_apply(lambda x: ((x.name+1)%7)+1, axis=1)
# all_trips_data["HourNum"] = all_trips_data.progress_apply(lambda x: int(x["StartTime"].split(":")[0]), axis=1)

# all_trips_data.to_csv("all_trips_LouVelo_recent.csv", index=False)

# def get_sorting_value(row):
#     time = float(row["StartTime"].replace(":", "."))
#     day = row["Day"]
#     month = row["Month"]
#     year = row["Year"]

#     return time+24*day+24*31*month+24*31*12*year

# tqdm.pandas()
# all_trips_data["Sort"] = all_trips_data.progress_apply(get_sorting_value, axis=1)

# all_trips_data = all_trips_data.sort_values(by="Sort")

# all_trips_data.to_csv("all_trips_LouVelo_recent.csv", index=False)