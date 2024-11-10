import logging
import numpy as np
import time
import cnst

from dataclasses import dataclass
from math import sqrt, pow, sin, cos, acos, tan, atan, log, pi
from mathmethod import fixed_point_iter


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#------ Abstraction ------#
@dataclass
class Celestial:
    mass: float = None
    mu: float = None
    grav_rad: float = None



@dataclass
class Orbit:
    r_a: float = None
    r_p: float = None
    a: float = None
    b: float = None
    e: float = None
    p: float = None
    i: float = None
    ta: float = None
    center_body: Celestial = None



#------ Implementation ------#
@dataclass
class EllipticOrbit(Orbit):
    period: float = None
    
    def __init__(self, center_body: Celestial, r_a: float, r_p: float):
        a = (r_a + r_p) / 2
        e = (r_a / a) - 1
        b = a * sqrt(1 - pow(e, 2))
        p = a * (1 - pow(e, 2))
        t = 2 * pi * sqrt(pow(a, 3) / center_body.mu)
        n = (2 * pi) / t
        
        self.r_a = r_a
        self.r_p = r_p
        self.a = a
        self.b = b
        self.e = e
        self.p = p
        self.center_body = center_body
        self.period = t
        self.n = n
        
    def calc_state(self, t: float, t_p: float = 0) -> np.ndarray:
        '''
        ### Метод расчёта положения на орбите и скорости движения в зависимости от времени
        
            t - время, на которое нужно определить положение   
            t_p - время прохождения перицентра   
        '''
        start_time = time.time()

        a = self.a
        e = self.e
        mu = self.center_body.mu
        p = self.p

        E_0 = (t - t_p) * self.n
        E_i = fixed_point_iter(lambda E_i: e * sin(E_i) + E_0, E_0)

        r = a * (1 - e * cos(E_i))   # радиус 
        ta = atan(sqrt((1 + e) / (1 - e)) * tan(E_i / 2)) * 2   # истинная аномалия

        x = r * cos(ta)
        y = r * sin(ta)

        spd_rate = sqrt(mu / p)
        rad_spd = spd_rate * e * sin(ta)
        trans_spd = spd_rate * (1 + e * cos(ta))
        
        vx = cos(ta) * rad_spd - sin(ta) * trans_spd
        vy = sin(ta) * rad_spd + cos(ta) * trans_spd
        
        logger.debug("--- Elliptic state: %.5f ms ---" % ((time.time() - start_time) * 1000))

        return np.array([x, y, vx, vy]).transpose()
    
    
    def calc_trajectory(self, dt: float, t_start: float = 0) -> np.ndarray:
        '''
        ### Метод расчёта траектории движения объекта по орбите
            dt: временной шаг траектории
            t_start: время, от которого нужно начать строить траекторию
        
        ##### Return:
            Траектория движения
        '''
        if t_start > self.period:
            raise ValueError("t_start не может быть больше периода обращения по орбите. t_start = " + t_start + ". period = " + self.period)
        
        steps = np.arange(t_start, self.period, dt)
        return np.array(list(map(lambda t: self.calc_state(t)[:2], steps)))

            
        
@dataclass
class HyperbolicOrbit(Orbit):
    v_exc: float = None
    
    def __init__(self, center_body: Celestial, r_p: float, v_exc: float):
        a = center_body.mu / pow(v_exc, 2)
        e = 1 + (r_p / a)
        p = a * (pow(e, 2) - 1)
        b = sqrt(p * a)
        
        self.r_p = r_p
        self.a = a
        self.b = b
        self.e = e
        self.p = p
        self.center_body = center_body
        
    def calc_flight_time(self) -> float:
        
        u_out = acos((self.p - Earth.grav_rad) / (Earth.grav_rad * self.e))
        sin_u_out, cos_u_out = sin(u_out), cos(u_out)
        
        e_spec = pow(self.e, 2) - 1
        ln_A = log(((self.e * sin_u_out + sqrt(e_spec)) / (1 + self.e * cos_u_out)) + (1 / sqrt(e_spec)))
        t_semi_trans = (sqrt(self.p / Earth.grav_rad) / e_spec) * ((self.e * self.p * sin_u_out / (1 + self.e * cos_u_out)) - (self.p * ln_A / sqrt(self.e - 1)))
        
        return t_semi_trans * 2



@dataclass
class Earth(Celestial):
    orbit: EllipticOrbit = None
    
    def __init__(self, kR = 1, kM = 1):
        '''
        ### Конструктор объекта
        
            kR: коэффициент масштабирования расстояния   
            kM: коэффициент масштабирования массы
        '''
        self.mass = 5.9722e+24 * kM
        self.mu = self.mass * cnst.G * (pow(kR, 3) / kM)
        self.grav_rad = 1.000e+09 * kR
        self.orbit = EllipticOrbit(center_body=Solar(kR, kM), r_a=1.521E+11 * kR, r_p=1.47095E+11 * kR)   
    
    @property    
    def period(self) -> float:
        return self.orbit.period
    
    @property
    def position(self, t: float, t_p: float = 0) -> np.ndarray:
        return self.orbit.calc_state(t, t_p)[:2]
    
    @property
    def speed(self, t: float, t_p: float = 0) -> np.ndarray:
        return self.orbit.calc_state(t, t_p)[2:4]
    
    def calc_trajectory(self, dt: float, t_start: float = 0) -> np.ndarray:
        return self.orbit.calc_trajectory(dt, t_start)



@dataclass        
class Solar(Celestial):
    
    def __init__(self, kR = 1, kM = 1):
        self.mass = 1.9885e+30 * kM
        self.mu = self.mass * cnst.G * (pow(kR, 3) / kM)
        

@dataclass
class Spacecraft:
    mass: float = None