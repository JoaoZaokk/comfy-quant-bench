def _sidecar_text(payload) -> str:
    """O texto do sidecar, com o fim de linha da plataforma.

    `os.linesep` porque os sidecars sempre foram gravados com `Path.write_text`, que em modo texto
    traduz LF para CRLF no Windows; manter isso deixa os bytes iguais aos de antes.
    """
    if callable(payload):
        payload = payload()
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return text.replace("\n", os.linesep)
