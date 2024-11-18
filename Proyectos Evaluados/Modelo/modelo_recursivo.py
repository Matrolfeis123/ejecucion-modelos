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



def optimizar_año(año, SoC_inicial, parametros_planta, CoD, generacion_list, df_cmg):

    # Desempaquetar los parámetros de la planta
    peak_power = parametros_planta['peak_power']
    nominal_power = parametros_planta['nominal_power']
    inverter_efficency_pv = parametros_planta['inverter_efficency_pv']
    degradacion_anual_pv = parametros_planta['degradacion_anual_pv']

    bess_charge_power = parametros_planta['bess_charge_power']
    bess_discharge_power = parametros_planta['bess_discharge_power']
    bess_charge_hours = parametros_planta['bess_charge_hours']
    bess_discharge_hours = parametros_planta['bess_discharge_hours']
    bess_initial_energy_capacity = parametros_planta['bess_initial_energy_capacity']
    bess_charge_efficency = parametros_planta['bess_charge_efficency']
    bess_discharge_efficency = parametros_planta['bess_discharge_efficency']
    degradacion_anual_bess = parametros_planta['degradacion_anual_bess']

    inverter_efficency_bess = parametros_planta['inverter_efficency_bess']
    carga_min_bess = parametros_planta['carga_min_bess']

    # Filtrar df de costos marginales para el año actual
    df_cmg_año = df_cmg[df_cmg['Año'] == año].reset_index(drop=True)
    costos_marginales = df_cmg_año['CMg'].tolist()

    # Definir T y t_indices
    T = len(costos_marginales)
    t_indices = range(T)
    print(T)

    # Verificar que 'generacion_list' tenga la longitud correcta
    if len(generacion_list) != T:
        print('Error: La longitud de los datos de generación no coincide con la de CMg')

        return None, None
    else:
        generacion_año = generacion_list


    # Calcular el indice de año desde el CoD
    indice_año = año - CoD

    # Calcular degradaciones acumuladas, considerando el año de augmentación de las baterías
    if año < CoD + parametros_planta['year_augmentation_bess']:
        degradacion_bess = (1 - degradacion_anual_bess) ** indice_año
        bess_actual_energy_capacity = bess_initial_energy_capacity * degradacion_bess

    else:
        #degradacion se reinicia y parte desde 0 nuevamente
        degradacion_bess = (1 - degradacion_anual_bess) ** (indice_año - parametros_planta['year_augmentation_bess'])
        bess_actual_energy_capacity = bess_initial_energy_capacity * degradacion_bess



    # Calcular Factores de degradacion acumulada: degradamos al inicio de cada ejecucion
    degradacion_acum_pv = (1-degradacion_anual_pv) ** indice_año
    
    # Definir G_pv_t con degradación acumulada: dado que G_pv_t es la generacion de los paneles, es esta variable la que degradamos, y asi el balance de energia se mantiene coherente.
    # G_pv_t = [generacion_año[t] * inverter_efficency_pv for t in t_indices]
    # Vamos a aplicar degradacion acumulada a todos los valores mayores que 0, mientras que si es menor a cero se mantiene (no pasa por el inversor ni se afecta por degradacion)
    G_pv_t = [generacion_año[t] * inverter_efficency_pv * degradacion_acum_pv if generacion_año[t] > 0 else generacion_año[t] for t in t_indices]
    

    # Definir el modelo de optimización
    model = pulp.LpProblem('Solar_PV_BESS_Optimization', pulp.LpMaximize)

    # Variables de decisión
    C_t = pulp.LpVariable.dicts('C_t', t_indices, lowBound=0, upBound=bess_charge_power, cat=pulp.LpContinuous)
    D_t = pulp.LpVariable.dicts('D_t', t_indices, lowBound=0, upBound=bess_discharge_power, cat=pulp.LpContinuous)
    SOC_t = pulp.LpVariable.dicts('SOC_t', t_indices, lowBound=0, upBound=bess_actual_energy_capacity, cat=pulp.LpContinuous)

    # Variables binarias para estado de carga y descarga
    charge_status = pulp.LpVariable.dicts("ChargeStatus", t_indices, cat='Binary')
    discharge_status = pulp.LpVariable.dicts("DischargeStatus", t_indices, cat='Binary')

    # Variables adicionales para inyección a la red y curtailment
    PV_grid_t = pulp.LpVariable.dicts('PV_grid_t', t_indices, lowBound=-0.01833, upBound=nominal_power, cat=pulp.LpContinuous)
    PV_curtail_t = pulp.LpVariable.dicts('PV_curtail_t', t_indices, lowBound=0, cat=pulp.LpContinuous)


    # Actualizar la función objetivo para incluir penalizaciones
    revenue_terms = []
    for t in t_indices:
        # Revenue from PV generation sold to the grid
        revenue_pv = PV_grid_t[t] * costos_marginales[t]
        # Revenue from BESS discharge
        revenue_bess = D_t[t] * costos_marginales[t]
        # Cost of charging the BESS (opportunity cost)
        cost_charge = C_t[t] * costos_marginales[t]
        # # Penalization for curtailment
        penalty_curtail = PV_curtail_t[t] * costos_marginales[t] if costos_marginales[t] > 0 else -0.1 * PV_curtail_t[t]

        # Total revenue at time t
        revenue_terms.append(revenue_pv + revenue_bess)

    # Establecer la función objetivo
    model += pulp.lpSum(revenue_terms)


    # 1. Balance de generación solar: Aca se soluciona lo del -0.015.
    # Ademas, dado que G_pv_t es la generacion de los paneles, es esta variable la que degradamos, y asi el balance de energia se mantiene coherente.
    for t in t_indices:
        model += (
            G_pv_t[t] == PV_grid_t[t] + C_t[t] + PV_curtail_t[t],
            f"PV_Generation_Balance_{t}"
        )

    # 2. Limitar la inyección neta al grid a la potencia nominal: Aca se soluciona lo del -0.015
    for t in t_indices:
        model += (
            PV_grid_t[t] + D_t[t] <= nominal_power,
            f"Max_Inyeccion_Neta_{t}"
        )
        model += (
            PV_grid_t[t] + C_t[t] <= peak_power,
            f"Max_Uso_PV_{t}"
        )
        # Debiera agregar el curtaiment a la restriccion de uso de PV, pero no se si es necesario, ya que el curtailment es una variable que se penaliza en la funcion objetivo y en el balance de energia ya se considera
        # por lo que no deberia ser necesario limitar su uso (?)

    # 3. Limites de carga y descarga
    for t in t_indices:
        if G_pv_t[t] >= 0:
            model += (
                C_t[t] <= G_pv_t[t],
                f"Battery_Charge_Limit_{t}"
            )
        else:
            model += (
                C_t[t] == 0,
                f"Battery_Charge_Zero_{t}"
            )

        model += (
            C_t[t] <= bess_charge_power,
            f"Max_Carga_{t}"
        )
        model += (
            D_t[t] <= bess_discharge_power,
            f"Max_Descarga_{t}"
        )

    # 4. Flujo del Estado de Carga (SoC)
    model += (
        SOC_t[0] == SoC_inicial + C_t[0] * bess_charge_efficency * inverter_efficency_bess - D_t[0] * (1 / (bess_discharge_efficency * inverter_efficency_bess)),
        'SOC_initial_condition'
    )

    for t in t_indices:
        if t > 0:
            model += (
                SOC_t[t] == SOC_t[t - 1] + 
                C_t[t] * bess_charge_efficency * inverter_efficency_bess - 
                D_t[t] * (1 / (bess_discharge_efficency * inverter_efficency_bess)),
                f"Estado_Carga_{t}"
            )
        
    # 5. Limites del SoC
    for t in t_indices:
        model += (
            SOC_t[t] <= bess_actual_energy_capacity, # Capacidad de almacenamiento actual, que se va actualizando cada año acorde a la degradacion
            f"Max_SOC_{t}"
        )
        model += (
            SOC_t[t] >= carga_min_bess,
            f"Min_SOC_{t}"
        )

    # 6. No simultaneidad de carga y descarga
    big_M = max(bess_charge_power, bess_discharge_power)
    for t in t_indices:
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

    # 7. Ventanas Horarias de Operación
    horas_carga_por_dia = int(bess_charge_hours)
    horas_descarga_por_dia = int(bess_discharge_hours)

    # Calcular las ventanas horarias
    ventanas_carga, ventanas_descarga = calcular_ventanas_carga_descarga_diario_v3(df_cmg_año, G_pv_t) # Antes, usabamos el vector de generacion sin degradacion. Ahora, utilizamos el vector con degradacion.

    # Añadir restricciones al modelo
    for t in t_indices:
        mes = df_cmg_año.iloc[t]['Mes']
        dia = df_cmg_año.iloc[t]['Día']
        hora = df_cmg_año.iloc[t]['Hora']

        # Obtener las horas de carga y descarga para el día actual
        horas_carga_dia = ventanas_carga.get((mes, dia), [])
        horas_descarga_dia = ventanas_descarga.get((mes, dia), [])

        # Restricciones de carga
        if hora in horas_carga_dia:
            pass
        else:
            model += (C_t[t] == 0, f"Horas_Carga_{t}")
            model += (charge_status[t] == 0, f"Estado_Carga_Horas_{t}")
        
        # Restricciones de descarga
        if hora in horas_descarga_dia:
            pass
        else:
            model += (D_t[t] == 0, f"Horas_Descarga_{t}")
            model += (discharge_status[t] == 0, f"Estado_Descarga_Horas_{t}")

    # Resolver el modelo
    solver = pulp.PULP_CBC_CMD(msg=True)
    model.solve(solver)

    print(f"Año {año}: {pulp.LpStatus[model.status]}")

    

    # Si la solución es óptima, extraer y analizar los resultados
    if pulp.LpStatus[model.status] == 'Optimal':
        C_sol = [C_t[t].varValue for t in t_indices]
        D_sol = [D_t[t].varValue for t in t_indices]
        SOC_sol = [SOC_t[t].varValue for t in t_indices]
        PV_grid_sol = [PV_grid_t[t].varValue for t in t_indices]
        PV_curtail_sol = [PV_curtail_t[t].varValue for t in t_indices]
        charge_status_sol = [1 if C_t[t].varValue > 0 else 0 for t in t_indices]
        discharge_status_sol = [1 if D_t[t].varValue > 0 else 0 for t in t_indices]


    

        results = pd.DataFrame()
        
        results['Año'] = df_cmg_año['Año'].values
        results['Mes'] = df_cmg_año['Mes'].values
        results['Dia'] = df_cmg_año['Día'].values
        results['Hora'] = df_cmg_año['Hora'].values


        #crearemos una columna que contenga el codigo del dato, el cual se arma concatenando el año, mes, dia y hora, separado por un _
        df_cmg_año['Codigo'] = df_cmg_año['Año'].astype(str) + '_' + df_cmg_año['Mes'].astype(str) + '_' + df_cmg_año['Día'].astype(str) + '_' + df_cmg_año['Hora'].astype(str) + '_MAITENCILLO'

        results['Codigo'] = df_cmg_año['Codigo'].values

        
        results_unir = pd.DataFrame({
            'Codigo': df_cmg_año['Codigo'].values,
            'carga': charge_status_sol,
            'descarga': discharge_status_sol,
            'SOC': SOC_sol,
            'CMg': [costos_marginales[t] for t in t_indices],
            'Generacion_PV': [G_pv_t[t] for t in t_indices],
            
            'PV_Inyectada_Grid': PV_grid_sol,
            'PV_Curtailment': PV_curtail_sol,
            'Carga_BESS': C_sol,
            'Descarga_BESS': D_sol,
            'inyeccion_neta': [PV_grid_sol[t] + D_sol[t] for t in t_indices],
            
            'Ingresos_BESS': [D_sol[t] * costos_marginales[t] for t in t_indices],
            'Ingresos_PV': [PV_grid_sol[t] * costos_marginales[t] for t in t_indices],
            'Ingresos_totales': [((D_sol[t] * costos_marginales[t]) + (PV_grid_sol[t] * costos_marginales[t])) for t in t_indices],
        })

        results = pd.merge(results, results_unir, on='Codigo', how='left')



        # # Calcular el despacho neto al grid
        # results['Despacho_Neto_Grid'] = results['PV_Inyectada_Grid'] + results['Descarga_BESS']

        # Extraer el SoC final
        SoC_final = SOC_sol[-1]

        return SoC_final, results

    else:
        print(f'La optimización no fue exitosa para el año {año}. Estado: {pulp.LpStatus[model.status]}')
        return None, None



# Parámetros de la planta
parametros_planta = {
    'peak_power': 30,  # MW
    'nominal_power': 30,  # MW
    'inverter_efficency_pv': 0.97,
    'degradacion_anual_pv': 0.0045,

    'bess_charge_power': 30,  # MW
    'bess_discharge_power': 30,  # MW
    'bess_charge_hours': 5,
    'bess_discharge_hours': 5,
    'bess_initial_energy_capacity': 150,  # MWh
    'degradacion_anual_bess': 0.02,
    'bess_charge_efficency': 0.92,
    'bess_discharge_efficency': 0.94,
    'inverter_efficency_bess': 0.97,
    'carga_min_bess': 0,
    'CoD': 2028,
    'year_augmentation_bess': 10 #Año a partir del cual se renuevan las baterias
}

CoD = parametros_planta['CoD']
vida_util_proyecto = 25  # Años

# Leer 'generacion.csv' una vez
generacion_df = pd.read_csv('Tatara/Modelo/Generaciones/generacion_tatara_pv_puro_27MWp.csv', sep=';')
generacion_list = generacion_df['G solar'].tolist()
for i in range(len(generacion_list)):
    generacion_list[i] = generacion_list[i].replace(',', '.')
    generacion_list[i] = max(-0.01666, float(generacion_list[i]))


# Estado inicial de carga (SoC)
SoC_inicial = 0  # Puede ser otro valor si se requiere

# Leer y formatear el archivo de costos marginales
path_csv = 'Modelo/CMg_tatara.csv'
df_cmg = formatear_df_cmg(path_csv)


resultados = pd.DataFrame()
for año in range(CoD, CoD + vida_util_proyecto):
    print(año)
    SoC_final, resultados_año = optimizar_año(año, SoC_inicial, parametros_planta, CoD, generacion_list, df_cmg)
    if resultados_año is not None:
        # Guardar resultados en un archivo
        resultados = pd.concat([resultados, resultados_año], ignore_index=True)
        # Actualizar SoC_inicial para el siguiente año
        SoC_inicial = SoC_final

    else:
        print(f"La optimización falló para el año {año}")
        breakpoint()
        break

# Guardar resultados en un archivo
try:
    resultados.to_excel('Modelo\Generaciones\output_pv_27MW_bess_90MWh_4.0hrs_manual.xlsx', index=False)

except:
    print('Error al guardar los resultados en el archivo, debes cerrar el archivo')
    breakpoint()
    resultados.to_excel('Modelo\Generaciones\output_pv_27MW_bess_90MWh_4.0hrs_manual.xlsx', index=False)

