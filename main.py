from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pickle
import pandas as pd
import numpy as np

# Load pre-trained models and data
preprocessor = pickle.load(open("pipeline.pkl", "rb"))
model = pickle.load(open("svm_model.pkl", "rb"))
df = pd.read_csv("data.csv")

# Define FastAPI app
app = FastAPI()

# Define Pydantic model for request
class Device(BaseModel):
    battery_power: int 	
    blue:bool
    clock_speed: float
    dual_sim:bool
    fc:int 
    four_g: bool
    int_memory:int
    m_dep:float
    mobile_wt:int
    n_cores:int
    pc:int
    px_height:int
    px_width:int
    ram:int
    sc_h:int
    sc_w:int
    talk_time: int
    three_g:bool
    touch_screen:bool
    wifi:bool 

class Prediction(BaseModel):
    price_range: int

# Database connection
conn = sqlite3.connect('devices.db')
c = conn.cursor()

# Create a devices table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS devices
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              battery_power INTEGER,
              blue INTEGER,
              clock_speed REAL,
              dual_sim INTEGER,
              fc INTEGER,
              four_g INTEGER,
              int_memory INTEGER,
              m_dep REAL,
              mobile_wt INTEGER,
              n_cores INTEGER,
              pc INTEGER,
              px_height INTEGER,
              px_width INTEGER,
              ram INTEGER,
              sc_h INTEGER,
              sc_w INTEGER,
              talk_time INTEGER,
              three_g INTEGER,
              touch_screen INTEGER,
              wifi INTEGER)''')

# Function to insert device data into database
def add_device_to_db(device):
    c.execute("INSERT INTO devices (battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (device.battery_power, device.blue, device.clock_speed, device.dual_sim, device.fc, device.four_g, device.int_memory, device.m_dep, device.mobile_wt, device.n_cores, device.pc, device.px_height, device.px_width, device.ram, device.sc_h, device.sc_w, device.talk_time, device.three_g, device.touch_screen, device.wifi))
    conn.commit()

# Function to get all devices from database
def get_all_devices_from_db():
    c.execute("SELECT * FROM devices")
    return c.fetchall()

# Function to get device by ID from database
def get_device_by_id_from_db(device_id):
    c.execute("SELECT * FROM devices WHERE id=?", (device_id,))
    return c.fetchone()

# Function to make prediction based on device ID
def predict_price_range(device_id):
    device_data = get_device_by_id_from_db(device_id)
    data = dict(zip(Device.__fields__, device_data[1:]))
    X = pd.DataFrame([data])
    if device_data:
        # Preprocess data and make prediction
        price_range = model.predict(preprocessor.transform(X))
        return {"price_range": int(price_range[0])}
    else:
        raise HTTPException(status_code=404, detail="Device not found")

# Define API endpoints
@app.get("/devices", response_model=list[Device])
async def get_all_devices():
    response =[]
    devices = get_all_devices_from_db()
    for device in devices: 
        response.append({"id": device[0], **dict(zip(Device.__fields__, device[1:]))})
    return response

@app.get("/devices/{device_id}", response_model=Device)
async def get_device(device_id: int):
    device = get_device_by_id_from_db(device_id)
    if device:
        return {"id": device[0], **dict(zip(Device.__fields__, device[1:]))}
    else:
        raise HTTPException(status_code=404, detail="Device not found")

@app.post("/devices", response_model=Device)
async def add_device(device: Device):
    add_device_to_db(device)
    return device

@app.get("/predict/{device_id}", response_model=Prediction)
async def predict_price(device_id: int):
    return predict_price_range(device_id)
