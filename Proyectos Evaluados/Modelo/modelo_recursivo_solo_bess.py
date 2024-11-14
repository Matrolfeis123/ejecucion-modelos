from matplotlib import pyplot as plt
import pulp
import pandas as pd
import datetime
from funciones_visualizacion_resultados import menu_graficos, menu_graficos_v2, grafico_curva_despacho_dia_navegable
from funciones_extra import calcular_ventanas_carga_descarga_año, calcular_ventanas_carga_descarga_diario, calcular_ventanas_carga_descarga_diario_v2, calcular_ventanas_carga_descarga_diario_v3


def formatear_df_cmg(path_csv: str):
    #COMPROBAR: Comprobar que en la ejecucion, se va recorriendo el csv y extrayendo los datos del año correspondiente año a año.
    # Cargar el CSV de costos marginales y filtrar por el año.
    costos_marginales_df = pd.read_csv(path_csv, sep=';', decimal=',') # Quizas el uso de memoria es mas eficiente si le entrego el df completo y que el haga el filtro cada ejecucion
    costos_marginales_df['Año'] = costos_marginales_df['Año'].astype(int)
    costos_marginales_df['Mes'] = costos_marginales_df['Mes'].astype(int)
    costos_marginales_df['Hora'] = costos_marginales_df['Hora'].astype(int)
    costos_marginales_df['CMg'] = costos_marginales_df['CMg'].astype(float)

    # Definir las fechas de inicio y fin para el año
    fecha_inicio = datetime.datetime(costos_marginales_df['Año'].min(), 1, 1, 0)
    fecha_fin = datetime.datetime(costos_marginales_df['Año'].max(), 12, 31, 23)
    rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='h')
    df_fechas = pd.DataFrame({'FechaHora': rango_fechas})

    # Extraer Año, Mes, Día y Hora
    df_fechas['Año'] = df_fechas['FechaHora'].dt.year
    df_fechas['Mes'] = df_fechas['FechaHora'].dt.month
    df_fechas['Día'] = df_fechas['FechaHora'].dt.day
    df_fechas['Hora'] = df_fechas['FechaHora'].dt.hour

    # Eliminar 29 de febrero si es necesario
    df_fechas = df_fechas[~((df_fechas['Mes'] == 2) & (df_fechas['Día'] == 29))]
    
    # Unir los datos de costos marginales al DataFrame de fechas
    df_cmg_año = pd.merge(df_fechas, costos_marginales_df, on=['Año', 'Mes', 'Hora'], how='left')

    # Verificar si hay valores nulos en CMg
    if df_cmg_año['CMg'].isnull().sum() > 0:
        print('Advertencia: CMg tiene valores nulos')

    # Optimizar tipos de datos
    df_cmg_año['Año'] = df_cmg_año['Año'].astype('int16')
    df_cmg_año['Mes'] = df_cmg_año['Mes'].astype('int8')
    df_cmg_año['Día'] = df_cmg_año['Día'].astype('int8')
    df_cmg_año['Hora'] = df_cmg_año['Hora'].astype('int8')
    df_cmg_año['CMg'] = df_cmg_año['CMg'].astype('float32')

    # Eliminar la columna 'FechaHora' si no es necesaria
    df_cmg_año.drop('FechaHora', axis=1, inplace=True)

    # Extraer la lista de costos marginales
    #Solo para probar, filtramos por año 2028
    # df_cmg_año = df_cmg_año[df_cmg_año['Año'] == año]
    # costos_marginales = df_cmg_año['CMg'].tolist()

    return df_cmg_año



def optimizar_año(año, SoC_inicial, parametros_planta, CoD, df_cmg):
    # Desempaquetar los parámetros de la planta
    bess_charge_power = parametros_planta['bess_charge_power']
    bess_discharge_power = parametros_planta['bess_discharge_power']
    bess_initial_energy_capacity = parametros_planta['bess_initial_energy_capacity']
    bess_charge_efficency = parametros_planta['bess_charge_efficency']
    bess_discharge_efficency = parametros_planta['bess_discharge_efficency']
    degradacion_anual_bess = parametros_planta['degradacion_anual_bess']
    inverter_efficency_bess = parametros_planta['inverter_efficency_bess']
    carga_min_bess = parametros_planta['carga_min_bess']
    year_augmentation_bess = parametros_planta['year_augmentation_bess']

    # Filtrar df de costos marginales para el año actual
    df_cmg_año = df_cmg[df_cmg['Año'] == año].reset_index(drop=True)
    costos_marginales = df_cmg_año['CMg'].tolist()

    # Definir T y t_indices
    T = len(costos_marginales)
    t_indices = range(T)
    print(T)

    # Calcular el índice de año desde el CoD
    indice_año = año - CoD

    # Calcular degradación acumulada
    if año < CoD + year_augmentation_bess:
        degradacion_bess = (1 - degradacion_anual_bess) ** indice_año
        bess_actual_energy_capacity = bess_initial_energy_capacity * degradacion_bess
    else:
        # Degradación se reinicia y parte desde 0 nuevamente
        degradacion_bess = (1 - degradacion_anual_bess) ** (indice_año - year_augmentation_bess)
        bess_actual_energy_capacity = bess_initial_energy_capacity * degradacion_bess

    # Definir el modelo de optimización
    model = pulp.LpProblem('BESS_Optimization', pulp.LpMaximize)

    # Variables de decisión
    C_t = pulp.LpVariable.dicts('C_t', t_indices, lowBound=0, upBound=bess_charge_power, cat=pulp.LpContinuous)
    D_t = pulp.LpVariable.dicts('D_t', t_indices, lowBound=0, upBound=bess_discharge_power, cat=pulp.LpContinuous)
    SOC_t = pulp.LpVariable.dicts('SOC_t', t_indices, lowBound=0, upBound=bess_actual_energy_capacity, cat=pulp.LpContinuous)

    # Variables binarias para estado de carga y descarga
    charge_status = pulp.LpVariable.dicts("ChargeStatus", t_indices, cat='Binary')
    discharge_status = pulp.LpVariable.dicts("DischargeStatus", t_indices, cat='Binary')

    # Función objetivo
    revenue_terms = []
    for t in t_indices:
        # Ingresos por venta de energía (descarga)
        revenue_bess = D_t[t] * costos_marginales[t]
        # Costos por compra de energía (carga)
        cost_charge = C_t[t] * costos_marginales[t]
        # Beneficio neto en el tiempo t
        net_revenue = revenue_bess - cost_charge
        revenue_terms.append(net_revenue)

    # Establecer la función objetivo
    model += pulp.lpSum(revenue_terms)

    # Restricciones
    # Flujo del SoC
    model += (
        SOC_t[0] == SoC_inicial + 
        C_t[0] * bess_charge_efficency * inverter_efficency_bess - 
        D_t[0] * (1/ (bess_discharge_efficency * inverter_efficency_bess)),
        'SOC_initial_condition'
    )

    for t in t_indices:
        if t > 0:
            model += (
                SOC_t[t] == SOC_t[t - 1] + 
                C_t[t] * bess_charge_efficency * inverter_efficency_bess - 
                D_t[t]*(1/ (bess_discharge_efficency * inverter_efficency_bess)),
                f"Estado_Carga_{t}"
            )

        # Límites del SoC
        model += (
            SOC_t[t] <= bess_actual_energy_capacity,
            f"Max_SOC_{t}"
        )
        model += (
            SOC_t[t] >= carga_min_bess,
            f"Min_SOC_{t}"
        )

        # Límites de carga y descarga
        model += (
            C_t[t] <= bess_charge_power,
            f"Max_Carga_{t}"
        )
        model += (
            D_t[t] <= bess_discharge_power,
            f"Max_Descarga_{t}"
        )

        # No simultaneidad de carga y descarga
        big_M = max(bess_charge_power, bess_discharge_power)
        model += (
            C_t[t] <= big_M * charge_status[t],
            f"Carga_Big_M_{t}"
        )
        model += (
            D_t[t] <= big_M * discharge_status[t],
            f"Descarga_Big_M_{t}"
        )
        model += (
            charge_status[t] + discharge_status[t] <= 1,
            f"Exclusividad_Carga_Descarga_{t}"
        )

    # Resolver el modelo
    solver = pulp.PULP_CBC_CMD(msg=True)
    model.solve(solver)

    print(f"Año {año}: {pulp.LpStatus[model.status]}")

    if pulp.LpStatus[model.status] == 'Optimal':
        C_sol = [C_t[t].varValue for t in t_indices]
        D_sol = [D_t[t].varValue for t in t_indices]
        SOC_sol = [SOC_t[t].varValue for t in t_indices]
        charge_status_sol = [charge_status[t].varValue for t in t_indices]
        discharge_status_sol = [discharge_status[t].varValue for t in t_indices]

        results = pd.DataFrame()
        results['Año'] = df_cmg_año['Año'].values
        results['Mes'] = df_cmg_año['Mes'].values
        results['Dia'] = df_cmg_año['Día'].values
        results['Hora'] = df_cmg_año['Hora'].values

        # Crear código identificador
        df_cmg_año['Codigo'] = df_cmg_año['Año'].astype(str) + '_' + df_cmg_año['Mes'].astype(str) + '_' + df_cmg_año['Día'].astype(str) + '_' + df_cmg_año['Hora'].astype(str) + '_BESS_PURO'
        results['Codigo'] = df_cmg_año['Codigo'].values

        # Crear el DataFrame con los resultados
        results_unir = pd.DataFrame({
            'Codigo': df_cmg_año['Codigo'].values,
            'carga': charge_status_sol,
            'descarga': discharge_status_sol,
            'Carga_BESS': C_sol,
            'Descarga_BESS': D_sol,
            'SOC': SOC_sol,
            'CMg': [costos_marginales[t] for t in t_indices],
            'Ingresos_BESS': [D_sol[t] * costos_marginales[t] for t in t_indices],
            'Costos_BESS': [C_sol[t] * costos_marginales[t] for t in t_indices],
            'Beneficio_Neto': [(D_sol[t] * costos_marginales[t]) - (C_sol[t] * costos_marginales[t]) for t in t_indices],
        })

        results = pd.merge(results, results_unir, on='Codigo', how='left')

        # Extraer el SoC final
        SoC_final = SOC_sol[-1]

        return SoC_final, results
    else:
        print(f'La optimización no fue exitosa para el año {año}. Estado: {pulp.LpStatus[model.status]}')
        return None, None



# Parámetros de la planta (solo BESS)
parametros_planta = {
    'bess_charge_power': 410,  # MW
    'bess_discharge_power': 410,  # MW
    'bess_initial_energy_capacity': 2460,  # MWh
    'degradacion_anual_bess': 0.02,
    'bess_charge_efficency': 0.92,
    'bess_discharge_efficency': 0.94,
    'inverter_efficency_bess': 0.97,
    'carga_min_bess': 0,
    'CoD': 2028,
    'year_augmentation_bess': 10  # Año de renovación de las baterías
}


CoD = parametros_planta['CoD']
vida_util_proyecto = 25  # Años

# Leer 'generacion.csv' una vez
# generacion_df = pd.read_csv('Modelo/Generaciones/generacion_tatara_pv_puro_33MWp.csv', sep=';')
# generacion_list = generacion_df['G solar'].tolist()
# for i in range(len(generacion_list)):
#     generacion_list[i] = generacion_list[i].replace(',', '.')
#     generacion_list[i] = max(100, float(generacion_list[i]))


# Estado inicial de carga (SoC)
SoC_inicial = 0  # Estado inicial de carga

# Leer y formatear el archivo de costos marginales
path_csv = 'Limar BESS Standalone\CMg_dongoyo.csv'
df_cmg = formatear_df_cmg(path_csv)

resultados = pd.DataFrame()
for año in range(CoD, CoD + vida_util_proyecto):
    print(año)
    SoC_final, resultados_año = optimizar_año(año, SoC_inicial, parametros_planta, CoD, df_cmg)
    if resultados_año is not None:
        # Guardar resultados
        resultados = pd.concat([resultados, resultados_año], ignore_index=True)
        # Actualizar SoC_inicial para el siguiente año
        SoC_inicial = SoC_final
    else:
        print(f"La optimización falló para el año {año}")
        breakpoint()
        break

# Guardar resultados en un archivo
try:
    resultados.to_excel(f'Limar BESS Standalone\output_bess_puro_{parametros_planta["bess_charge_power"]}MWp_{parametros_planta["bess_initial_energy_capacity"]/parametros_planta["bess_charge_power"]}hrs.xlsx', index=False)
except:
    print('Error al guardar los resultados en el archivo, debes cerrar el archivo')
    breakpoint()
    resultados.to_excel(f'Limar BESS Standalone\output_bess_puro_{parametros_planta["bess_charge_power"]}MWp_{parametros_planta["bess_initial_energy_capacity"]/parametros_planta["bess_charge_power"]}hrs.xlsx', index=False)
