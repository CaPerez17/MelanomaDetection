# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

# Definir los parámetros del modelo
beta = 0.02 # Tasa de infección
gamma = 0.01 # Tasa de recuperación
N = 10000 # Población total
S0 = 10000 # Susceptibles iniciales
I0 = 10 # Infectados iniciales

# Definir el intervalo de tiempo y el paso
t0 = 0 # Tiempo inicial
tf = 100 # Tiempo final
dt = 0.001 # Paso de tiempo
t = np.arange(t0, tf + dt, dt) # Vector de tiempo

# Definir la función del sistema SIS
def sis(S, I):
    dSdt = -beta * S * I + gamma * I # Ecuación para susceptibles
    dIdt = beta * S * I - gamma * I # Ecuación para infectados
    return dSdt, dIdt

# Crear vectores vacíos para guardar los resultados
S = np.zeros(len(t)) # Vector para susceptibles
I = np.zeros(len(t)) # Vector para infectados

# Asignar las condiciones iniciales
S[0] = S0 
I[0] = I0 
# Usar el método de Euler para resolver el sistema numéricamente
for i in range(1, len(t)):
    dSdt, dIdt = sis(S[i-1], I[i-1]) # Calcular las derivadas en el paso anterior
    S[i] = S[i-1] + dt * dSdt # Actualizar el valor de susceptibles en el paso actual
    I[i] = I[i-1] + dt * dIdt # Actualizar el valor de infectados en el paso actual

# Graficar los resultados
plt.plot(t, S, label='Susceptibles')
plt.plot(t, I, label='Infectados')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.show()