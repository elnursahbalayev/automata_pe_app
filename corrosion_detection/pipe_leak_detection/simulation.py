import random
import time
import math
import pandas as pd

class DataSimulator:
    def __init__(self):
        self.pressure = 100.0  # Initial pressure in psi
        self.flow_rate = 5000.0 # Initial flow in bbl/day
        self.leak_active = False
        self.leak_start_time = 0
        self.prior_prob = 0.01
        self.posterior_prob = 0.01
        self.thermal_anomaly_detected = False

    def toggle_leak(self, active):
        """Manually trigger or stop a leak event."""
        self.leak_active = active
        if active:
            self.leak_start_time = time.time()
            self.thermal_anomaly_detected = True
        else:
            self.thermal_anomaly_detected = False
            self.pressure = 100.0 # Reset
            self.posterior_prob = 0.01

    def update(self):
        """Update system state for the next time step."""
        
        # 1. Simulate Pressure Physics
        noise = random.gauss(0, 0.5)
        
        if self.leak_active:
            # Exponential decay of pressure during leak
            elapsed = time.time() - self.leak_start_time
            decay = 20 * (1 - math.exp(-0.2 * elapsed)) # Drops up to 20 psi
            target_pressure = 100.0 - decay
            
            # Add turbulence (more noise during leak)
            noise = random.gauss(0, 2.0)
            self.pressure = target_pressure + noise
            
            # Flow rate drops slightly
            self.flow_rate = 5000.0 - (decay * 10) + random.gauss(0, 50)
            
        else:
            # Normal operation: Sine wave fluctuation
            self.pressure = 100.0 + math.sin(time.time() / 5) * 2 + noise
            self.flow_rate = 5000.0 + random.gauss(0, 20)

        # 2. Bayesian Update Logic
        # P(Leak | Data) = P(Data | Leak) * P(Leak) / P(Data)
        
        # Likelihoods (Simplified)
        p_low_pressure_given_leak = 0.9
        p_low_pressure_given_normal = 0.05
        
        p_visual_given_leak = 0.95
        p_visual_given_normal = 0.01
        
        # Current Evidence
        is_pressure_low = self.pressure < 90.0
        is_visual_detection = self.thermal_anomaly_detected
        
        # Update Prior based on evidence
        current_prob = self.posterior_prob
        
        # Pressure Update
        if is_pressure_low:
            numerator = p_low_pressure_given_leak * current_prob
            denominator = (p_low_pressure_given_leak * current_prob) + (p_low_pressure_given_normal * (1 - current_prob))
            current_prob = numerator / denominator
            
        # Visual Update
        if is_visual_detection:
            numerator = p_visual_given_leak * current_prob
            denominator = (p_visual_given_leak * current_prob) + (p_visual_given_normal * (1 - current_prob))
            current_prob = numerator / denominator
            
        # Decay if no evidence (return to normal)
        if not is_pressure_low and not is_visual_detection:
            current_prob = max(0.01, current_prob * 0.9)
            
        self.posterior_prob = current_prob

        return {
            "pressure": round(self.pressure, 2),
            "flow_rate": int(self.flow_rate),
            "probability": round(self.posterior_prob, 4),
            "leak_active": self.leak_active
        }

# Global instance
simulator = DataSimulator()
