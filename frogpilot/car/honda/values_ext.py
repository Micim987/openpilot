from enum import IntFlag

from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car import CarSpecs, Platforms, dbc_dict
from openpilot.selfdrive.car.honda.values import HondaCarDocs, HondaNidecPlatformConfig


class HondaFlagsFP(IntFlag):
  EPS_MODIFIED = 128


class HondaFlagsRP(IntFlag):
  NIDEC_PEDAL_TUNE = 1
  NIDEC_PEDAL_DEADZONE = 2


class CAR_RP(Platforms):
  # Nidec Cars
  HONDA_CLARITY = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Clarity 2018-22", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=4052. * CV.LB_TO_KG, wheelbase=2.75, centerToFrontRatio=0.41, steerRatio=16.50, tireStiffnessFactor=1.),
    dbc_dict('honda_clarity_hybrid_2018_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlagsRP.NIDEC_PEDAL_TUNE,
  )
  HONDA_ODYSSEY = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Odyssey 2018-20")],
    CarSpecs(mass=1900, wheelbase=3.0, steerRatio=14.35, centerToFrontRatio=0.41, tireStiffnessFactor=0.82),
    dbc_dict('honda_odyssey_exl_2018_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlagsRP.NIDEC_PEDAL_DEADZONE,
  )
  HONDA_PILOT = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Honda Pilot 2016-22", min_steer_speed=12. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Passport 2019-23", "All", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=4278 * CV.LB_TO_KG, wheelbase=2.86, centerToFrontRatio=0.428, steerRatio=16.0, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlagsRP.NIDEC_PEDAL_DEADZONE,
  )
  HONDA_RIDGELINE = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Ridgeline 2017-24", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=4515 * CV.LB_TO_KG, wheelbase=3.18, centerToFrontRatio=0.41, steerRatio=15.59, tireStiffnessFactor=0.444),  # as spec
    dbc_dict('acura_ilx_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlagsRP.NIDEC_PEDAL_DEADZONE,
  )
  HONDA_CIVIC = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Civic 2016-18", min_steer_speed=12. * CV.MPH_TO_MS, video_link="https://youtu.be/-IkImTe1NYE")],
    CarSpecs(mass=1326, wheelbase=2.70, centerToFrontRatio=0.4, steerRatio=15.38),  # 10.93 is end-to-end spec
    dbc_dict('honda_civic_touring_2016_can_generated', 'acura_ilx_2016_nidec'),
    flags=HondaFlagsRP.NIDEC_PEDAL_TUNE,
  )


HONDA_NIDEC_PEDAL_TUNE = CAR_RP.with_flags(HondaFlagsRP.NIDEC_PEDAL_TUNE)
HONDA_NIDEC_PEDAL_DEADZONE = CAR_RP.with_flags(HondaFlagsRP.NIDEC_PEDAL_DEADZONE)