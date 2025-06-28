import numpy as np
import math

def velocity_function(caudal, diameter):
    return (4 * caudal) / (np.pi * (diameter**2))

def reynolds_number(velocity, diameter, density, viscosity):
    return (density * velocity * diameter) / viscosity

'''-------------------------- Cálculo de pérdidas por fricción ---------------------------'''

def friction_pressure_losses(f, length, diameter, density, velocity):
    """
    Calculate pressure losses in a pipe using the Darcy-Weisbach equation.
    
    Parameters:
    f (float): Darcy friction factor
    length (float): Length of the pipe (m)
    diameter (float): Diameter of the pipe (m)
    density (float): Density of the fluid (kg/m^3)
    velocity (float): Velocity of the fluid (m/s)
    
    Returns:
    float: Pressure loss (Pa)
    """
    return f * (length / diameter) * 0.5 * density * (velocity ** 2)

def swamee_jain(Reynolds, relative_roughbess):
    """
    Calculate the friction factor using the Swamee-Jain equation.
    
    Parameters:
    Reynolds (float): Flow Reynolds number
    Relative rughness (float): Pipe relative roughness
    
    Returns:
    float: Friction factor
    """

    if Reynolds < 2300:
        # For laminar flow, use the formula f = 64 / Re
        f = 64 / Reynolds
    else:
        # Calculate friction factor using Swamee-Jain equation
        f = 0.25 / ( ( ( np.log10( ((relative_roughbess) / 3.7) + (5.74 / (Reynolds ** 0.9))) ) ) ** 2)
    
    return f

'''------------------------ Cálculos de pérdidas por singularidad -------------------------'''

def singularity_pressure_losses(velocity, density, k):
    """
    Calculate pressure losses due to singularities (valves, bends, etc.).
    
    Parameters:
    velocity (float): Fluid velocity in m/s
    density (float): Fluid density in kg/m^3
    diameter (float): Pipe diameter in m
    k (float): Loss coefficient for the singularity
    
    Returns:
    float: Pressure loss (Pa)
    """
    return k * 0.5 * density * (velocity**2)

fitting_constants_3k = {
    "90° elbow, standard, threaded": [800, 0.140, 4.0],
    "90° elbow, standard, flanged": [800, 0.091, 4.0],
    "90° elbow, long radius, threaded": [800, 0.071, 4.2],
    "90° mitered elbow, 1 weld (90° angle)": [1000, 0.270, 4.0],
    "90° mitered elbow, 2 welds (45° angle)": [800, 0.068, 4.1],
    "90° mitered elbow, 3 welds (30° angle)": [800, 0.035, 4.2],
    "45° elbow, standard, all types": [500, 0.071, 4.2],
    "45° elbow, long radius, all types": [500, 0.052, 4.0],
    "45° mitered elbow, 1 weld (45° angle)": [500, 0.086, 4.0],
    "45° mitered elbow, 2 welds (22½° angle)": [500, 0.052, 4.0],
    "180° bend, standard, threaded": [1000, 0.230, 4.0],
    "180° bend, standard, flanged or welded": [1000, 0.120, 4.0],
    "180° bend, long radius, all types": [1000, 0.100, 4.0],
    "Valve, gate β = d/D = 1.0": [300, 0.037, 3.9],
    "Valve, globe, standard": [1500, 1.700, 3.6],
    "Valve, 90° angle": [1000, 0.690, 4.0],
    "Valve, 45° angle": [950, 0.250, 4.0],
    "Valve, ball": [300, 0.017, 3.6],
    "Valve, butterfly": [1000, 0.690, 4.9],
    "Valve, check, lift": [2000, 2.850, 3.8],
    "Valve, check, swing": [1500, 0.460, 4.0],
    "Entrance, flush square (no pipe projection)": [0, 0.50, 0],
    "Entrance, flush rounded r/D = 0.02": [0, 0.28, 0],
    "Entrance, flush rounded r/D = 0.04": [0, 0.24, 0],
    "Entrance, flush rounded r/D = 0.06": [0, 0.15, 0],
    "Entrance, flush rounded r/D = 0.10": [0, 0.09, 0],
    "Entrance, flush rounded r/D ≤ 0.15": [0, 0.04, 0],
    "Entrance, chamfered": [0, 0.25, 0],
    "Entrance, pipe projecting inward": [0, 0.78, 0],
    "Exit, all pipe exits": [0, 1.00, 0],
}

def method_3k(Reynolds, nominal_diameter, fitting_type):
    """
    Calculate singularity coefficient using the 3k method.
    
    Parameters:
    Reynolds (float): Reynolds number of the fluid
    Nominal diameter (float): Nominal diameter of the pipe (in)
    Fitting type (string): Name of the fitting pipe
    
    Returns:
    K: Singularity coefficient
    """

    K1, K_infinity, KD = fitting_constants_3k[fitting_type]
    return (K1 / Reynolds) + (K_infinity * (1 + (KD / ((nominal_diameter) ** 0.3))))

def K_converging_fitting(type, Qbranch, Qcombined, Dbranch, Dcombined, alpha):
    '''Calculate the K value in a converging fitting.
    
    Parameters:
    type (str): Type of fitting, either "branch" or "run".
    Qbranch (float): Flow rate in the branch (m³/s).
    Qcombined (float): Combined flow rate (m³/s).
    Dbranch (float): Diameter of the branch (m).
    Dcombined (float): Diameter of the combined pipe (m).
    alpha (int): Angle of the fitting in degrees (30, 45, 60, or 90).
    
    Returns:
    float: K value for the fitting.
    '''
    # Parámetros base
    yb = Qbranch / Qcombined
    beta_b = Dbranch / Dcombined

    # Constantes según el ángulo
    table_4_12 = {
        30: {"C": "tabla_413", "D": 1, "E": 2, "F": 1.74},
        45: {"C": "tabla_413", "D": 1, "E": 2, "F": 1.41},
        60: {"C": "tabla_413", "D": 1, "E": 2, "F": 1},
        90: {"C": "tabla_413", "D": 1, "E": 2, "F": 0}
    }

    consts = table_4_12[alpha]
    D = consts["D"]
    E = consts["E"]
    F = consts["F"]

    # Valor de C (tabla 4.13)
    if beta_b ** 2 <= 0.35 and yb <= 0.35:
        C_branch = 1
    elif beta_b ** 2 <= 0.35 and yb > 0.35:
        C_branch = 1
    elif beta_b ** 2 > 0.35 and yb <= 0.35:
        C_branch = 0.9 * (1 - yb)
    else: 
        C_branch = 0.55

    if type == "branch":

        K_branch = C_branch * (1 + (D * ( (yb / (beta_b ** 2)) ** 2)) - (E * ((1 - yb) ** 2)) - (F * ((yb / beta_b) ** 2)))
        return K_branch
    
    elif type == "run":

        # K_run
        if alpha == 90:
            K_run = 1.55 * yb - yb**2
        else:
            C_run = 1
            D_run = 0
            E_run = 1
            F_run = table_4_12[alpha]["F"]
            K_branch = C_run * (1 + (D_run * ( (yb / (beta_b ** 2)) ** 2)) - (E_run * ((1 - yb) ** 2)) - (F_run * ((yb / beta_b) ** 2)))
            return K_run

def K_diverging_fitting(type, Qbranch, Qcombined, Dbranch, Dcombined, alpha):
    '''Calculate the K value in a diverging fitting.
    
    Parameters:
    type (str): Type of fitting, either "branch" or "run".
    Qbranch (float): Flow rate in the branch (m³/s).
    Qcombined (float): Combined flow rate (m³/s).
    Dbranch (float): Diameter of the branch (m).
    Dcombined (float): Diameter of the combined pipe (m).
    alpha (int): Angle of the fitting in degrees (30, 45, 60, or 90).
    
    Returns:
    float: K value for the fitting.
    ''' 
    yb = Qbranch / Qcombined
    beta_b = Dbranch / Dcombined
    beta_b_sq = beta_b ** 2
    alpha_rad = math.radians(alpha)

    if type == "branch":

        # G según tabla 4.14 y 4.15
        if alpha <= 60:
            if beta_b_sq <= 0.35:
                if yb <= 0.6:
                    G = 1.1 - (0.7 * yb)
                else:
                    G = 0.85
            else: # if beta_b_sq > 0.35:
                if yb <= 0.4:
                    G = 1.0 - (0.6 * yb)
                else:
                    G = 0.60
            H = 1
            J = 2

        elif alpha == 90:

            if beta_b <= 0.67:
                G = 1
                H = 1
                J = 2
            else:
                G = 1 + (0.3 * (yb ** 2))
                H = 0.3
                J = 0

        # K_branch (Ecuación 4.24)
        try:
            term = (yb / beta_b_sq)
        except UnboundLocalError:
            print(f"Error: yb = {yb}, beta_b = {beta_b}")
        # term = (yb / beta_b_sq)
        try:
            K_branch = G * (1 + (H * (term ** 2)) - (J * term * math.cos(alpha_rad)))
        except UnboundLocalError:
            print(f"Error: yb = {yb}, beta_b = {beta_b}")

        return K_branch

    elif type == "run":

        # M según tabla 4.16
        if beta_b_sq <= 0.4:
            M = 0.4
        else: # if beta_b_sq > 0.4
            if yb <= 0.5:
                M = 2 * ((2 * yb) - 1)
            else:
                M = 0.3 * ((2 * yb) - 1)

        # K_run (Ecuación 4.25)
        K_run = M * (yb ** 2)

        return K_run
    
def K_square_reduction(before_reynolds, before_f, before_diameter, afer_diameter):
    beta = afer_diameter / before_diameter
    if before_reynolds < 2500:
        K = (1.2 + (160 / before_reynolds)) * ((1 / (beta ** 4)) - 1)
    else:
        K = (0.6 + (0.48 * before_f)) * (beta ** (-2)) * ((beta ** (-2) - 1))

    return K

def K_rounded_reduction(before_reynolds, before_f, before_diameter, afer_diameter):
    beta = afer_diameter / before_diameter
    K = (0.1 + (50 / before_reynolds)) * ((1 / (beta ** 4)) - 1)
    return K