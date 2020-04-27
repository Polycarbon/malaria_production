import platform
import netifaces as ni
import ifaddr

def get_ip_linux(interface='eth0'):
    """ 
    interface list: 'lo', 'eth0', 'wlan0', 'eth3', 'vboxnet0'  
    """
    if interface not in ni.interfaces(): return None
    ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']
    return ip

def get_ip_window(interface='eth',verbose=False):
    """ 
    interface list: 'wifi' , 'eth'
    """
    adapters = ifaddr.get_adapters()
    for adapter in adapters:
        if "Virtual".lower() in adapter.nice_name.lower(): continue
        if interface == 'eth' and "Ethernet Adapter".lower() in adapter.nice_name.lower():
            if verbose:
                print("IPs of network adapter " + adapter.nice_name)
                print( "   %s/%s" % (adapter.ips[0].ip, adapter.ips[0].network_prefix))
            return adapter.ips[0].ip
        elif interface == 'wifi' and "Wireless Network Adapter".lower() in adapter.nice_name.lower():
            if verbose:
                print("IPs of network adapter " + adapter.nice_name)
                print( "   %s/%s" % (adapter.ips[0].ip, adapter.ips[0].network_prefix))
            return adapter.ips[0].ip
    return None

def get_adapter_list():
    adapters = ifaddr.get_adapters()
    for adapter in adapters:
        print("IPs of network adapter " + adapter.nice_name)
        for ip in adapter.ips:
            print( "   %s/%s" % (ip.ip, ip.network_prefix))

def get_ip(os_name = None,*args, **kwargs):
    if os_name is None:
        os_name = platform.system()
    if os_name == 'Windows': return get_ip_window(*args, **kwargs)
    else: return get_ip_linux(*args, **kwargs)

# Example
# print(get_ip(interface='wifi',verbose=True))
