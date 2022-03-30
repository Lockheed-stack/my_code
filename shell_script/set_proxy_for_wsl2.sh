#!/bin/bash

printf "Setting proxy and valid only for this login\n"
host_ip=$(cat /etc/resolv.conf |grep nameserver|cut -d " " -f 2)
export ALL_PROXY="${host_ip}:1080"

echo "Testing for visiting google"
status=$(curl -Is www.google.com -m 5|grep HTTP|cut -d " " -f 2)

if test -n "$status"
    then echo "Test of visiting google: Passed"
else
    echo "timeout, please check your configuration"
fi