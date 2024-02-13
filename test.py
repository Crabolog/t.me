import datetime
event_start = datetime.datetime.strptime('2024-02-12 19:43:55.985354', '%Y-%m-%d %H:%M:%S.%f')
event_end = datetime.datetime.now()
event_start = int(event_start.strftime('%Y%m%d'))
event_end = int(event_end.strftime('%Y%m%d'))
x = event_end-event_start
# abs((event_end - event_start).days)#.strftime('%Y-%m-%d %H:%M:%S.%f')
print(event_start)
print(event_end)
print(x)
 
