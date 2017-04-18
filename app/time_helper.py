import time

class TimeHelper():

  def __init__(self):
    self.timer_start = self.current_time = self.get_current_time_ms()
  
  def get_current_time_ms(self):
    self.current_time = int(round(time.time()*1000))
    return self.current_time

  def diff_ms(self, time):
    diff = abs(time - self.timer_start)
    self.timer_start = self.get_current_time_ms()

    return diff

  def ms_to_s(self, time):
    return round(time/1000, 3)

  def s_format(self, time):
    seconds = int(time % 60)
    minutes = int(time / 60 % 60)
    hours = int(time / 60 / 60 % 60)

    return "%s:%s:%s" % (str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2))