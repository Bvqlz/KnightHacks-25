from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy 
import json

app = Flask(__name__)
CORS(app)

#Import data to be used
def initialize_data():
    try:
        asset_indexes = numpy.load('asset_indexes.npy')
        distance_matrix = numpy.load('distance_matrix.npy')
        photo_indexes = numpy.load('photo_indexes.npy')
        points_lat_lon = numpy.load('points_lat_long.npy')
        predecessors = numpy.load('predecessors.npy')
        waypoint_indexes = numpy.load('waypoint_indexes.npy')
        return {
            'asset_indexes' : asset_indexes,
            'photo_indexes' : photo_indexes,
            'distance_matrix' : distance_matrix,
            'points_lat_lon' : points_lat_lon,
            'predecessors' : predecessors,
            'waypoint_indexes' : waypoint_indexes
        }
    except FileNotFoundError as e:
        print(f"Data loading Error: {e}")
        return None
    
data = initialize_data()

def getCoords(pathway):
    points=data['points_lat_lon']
    pathway
    coords = points[pathway]
    for i in range (len(coords)): #flips lat and lon to be in correct orientation
        tempvar = coords[i][0]
        coords[i][0] = coords[i][1]
        coords[i][1] = tempvar
    coords = str(coords)
    return coords

pathsend = [10,7,2,4,17]

@app.route('/coords')
def display():
    return getCoords([1,17,8,2])

@app.route('/')
@app.route('/mission')
def displayOutput():
    with open('output/asset_points.json', 'r') as file:
        asset_points = json.load(file)
    file.close()
    with open('output/mission_paths.json','r') as file:
        mission_paths = json.load(file)
    file.close()
    with open('output/photo_points.json','r') as file:
        photo_points = json.load(file)
    file.close()
    with open('output/polygon_boundary.json','r') as file:
        polygon_boundary = json.load(file)
    file.close()
    jsonFinal = asset_points | mission_paths | photo_points | polygon_boundary
    return jsonFinal