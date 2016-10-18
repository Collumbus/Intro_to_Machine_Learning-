#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here

    # Calculando erro residual e inserindo os dados em uma lista de tuplas
    cleaned_data = zip(ages, net_worths, abs(predictions - net_worths))

    # Ordeando a lisa em ordem crescente de acordo co o erro
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2], reverse=False)

    #Eliminando 10% dos pontos onde ha maior erro residual
    cleaned_data = list( cleaned_data[:int( len(ages) - len(ages)/10 )] )

    return cleaned_data
