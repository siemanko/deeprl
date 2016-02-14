DEFAULT_SETTINGS = {
    'queues': {
        'type': 'process'
    }
}

def update_settings(original, updates, key_name="settings"):
    # if one piece is undefined return the other
    if original is None:
        return updates
    if updates is None:
        return original

    # if they are both dictionaries merge them
    if isinstance(original, dict):
        assert isinstance(original, dict), "Expected %s to be a dictionary." % (key_name,)
        res = {}
        for key in list(original.keys()) + list(updates.keys()):
            res[key] = update_settings(original.get(key), updates.get(key), key_name=key)
        return res

    # if both are defined and non-dictionaries, updates have priority
    return updates
