imulacoesArtificiais/simulacoes3C/simulacao3D_49.npz"
    )

    print("estado shape:", dados["estado"].shape)
    print("tempo shape:", dados["tempo"].shape)

    print("\nESTADO[0]:")
    print(dados["estado"][0])

    print("\nTEMPO[0:10]:")
    print(dados["tempo"][:10])

    print("\nÚLTIMA COLUNA DO ESTADO:")
    print(dados["estado"][:10, -1])