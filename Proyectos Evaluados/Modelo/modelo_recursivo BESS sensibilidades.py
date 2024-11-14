from matplotlib import pyplot as plt
import pulp
import pandas as pd
import datetime
from modelo_recursivo_solo_bess import optimizar_año


def sensibilidad_bess_initial_energy_capacity(parametros_planta, SoC_inicial, CoD, vida_util_proyecto, df_cmg, capacidad_values):
    resultados_sensibilidad = {}
    resumen_beneficios = []

    for capacidad in capacidad_values:
        print(f"\nEjecución de sensibilidad para bess_initial_energy_capacity = {capacidad} MWh")
        # Actualizar el parámetro de capacidad inicial en los parámetros de la planta
        parametros_planta['bess_initial_energy_capacity'] = capacidad

        # Ejecución de la optimización para el horizonte de 25 años
        resultados = pd.DataFrame()
        beneficio_total = 0

        for año in range(CoD, CoD + vida_util_proyecto):
            print(f"Año {año}")
            SoC_final, resultados_año = optimizar_año(año, SoC_inicial, parametros_planta, CoD, df_cmg)
            if resultados_año is not None:
                # Acumular resultados y beneficio neto anual
                resultados = pd.concat([resultados, resultados_año], ignore_index=True)
                beneficio_anual = resultados_año['Beneficio_Neto'].sum()
                beneficio_total += beneficio_anual
                
                # Actualizar SoC_inicial para el siguiente año
                SoC_inicial = SoC_final
            else:
                print(f"La optimización falló para el año {año}")
                break

        # Guardar los resultados de la optimización con la capacidad actual
        resultados_sensibilidad[capacidad] = resultados

        # Añadir resumen de beneficios netos al resumen tabulado
        resumen_beneficios.append({
            'Capacidad Inicial (MWh)': capacidad,
            'Beneficio Neto Total': beneficio_total
        })

        # Guardar los resultados de la ejecución en un archivo
        try:
            resultados.to_excel(f'output_bess_sensibilidad_{capacidad}MWh.xlsx', index=False)
        except:
            print('Error al guardar los resultados en el archivo, debes cerrar el archivo')
            resultados.to_excel(f'output_bess_sensibilidad_{capacidad}MWh.xlsx', index=False)

    # Convertir el resumen de beneficios a un DataFrame para visualización
    resumen_df = pd.DataFrame(resumen_beneficios)
    print("\nResumen de beneficios netos para cada capacidad inicial:")
    print(resumen_df)
    
    return resultados_sensibilidad, resumen_df
