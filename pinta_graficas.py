# -*- coding: utf-8 -*-
"""
Created on Fri Sep 05 10:33:42 2014
@author: navia
"""
import pylab as pl

pl.clf()

size = [5, 10, 20, 50, 75, 100, 125, 150, 200, 250, 300]

adult_auc = [0.8039, 0.8638, 0.8776, 0.8996, 0.8913, 0.8914, 0.8988, 0.8987, 0.8976, 0.8992, 0.8978]
susy_auc = [0.6365, 0.7943, 0.8297, 0.8602, 0.8648, 0.8685, 0.8682, 0.8658]
higgs_auc = [0.5375, 0.54, 0.566, 0.6057, 0.6445, 0.657]
kddcup_auc = [0.7481, 0.9452, 0.964, 0.9756, 0.9515, 0.9513, 0.9755, 0.9517]

pl.plot(size, adult_auc, 'ko-', label='Adult')
pl.plot(size[0:len(susy_auc)], susy_auc, 'r^-', label='Susy')
pl.plot(size[0:len(higgs_auc)], higgs_auc, 'gs-', label='Higgs')
pl.plot(size[0:len(kddcup_auc)], kddcup_auc, 'm*-', label='Kdd-99')

#pl.plot([0, 1], [0, 1], 'k--')
#pl.plot([1, 0], [0, 1], 'k--')
#pl.plot([0.05, 0.05], [0, 1], 'k--')
#pl.plot([0, 1] ,[0.95, 0.95], 'k--')
pl.xlim([0.0, 300.0])
pl.ylim([0.5, 1.0])
pl.xlabel('Size')
pl.ylabel('AUC')
#pl.title('Curva ROC')
pl.legend(loc="lower right")
pl.grid()

pl.savefig('aucs.eps', format='eps', dpi=600)

pl.show()

adult_time = [3.238203125, 5, 11.5215625, 21.34351563, 25, 35.47710938, 55.34, 65.39, 70, 83.1953125, 90.63]
susy_time = [7.531953125, 9, 11.00273438, 26.81804688, 50, 74.12085938, 90, 110]
higgs_time = [5.098828125, 5.63671875, 15, 43.56367188, 73.9234375, 98]
kddcup_time = [0.901796875, 5.635625, 10.08046875, 19.98125, 34.104375, 45, 69, 93]

pl.plot(size, adult_time, 'ko-', label='Adult')
pl.plot(size[0:len(susy_time)], susy_time, 'r^-', label='Susy')
pl.plot(size[0:len(higgs_time)], higgs_time, 'gs-', label='Higgs')
pl.plot(size[0:len(kddcup_time)], kddcup_time, 'm*-', label='Kdd-99')

#pl.plot([0, 1], [0, 1], 'k--')
#pl.plot([1, 0], [0, 1], 'k--')
#pl.plot([0.05, 0.05], [0, 1], 'k--')
#pl.plot([0, 1] ,[0.95, 0.95], 'k--')
pl.xlim([0.0, 300.0])
pl.ylim([0.5, 140])
pl.xlabel('Size')
pl.ylabel('Time (min.)')
#pl.title('Curva ROC')
pl.legend(loc="lower right")
pl.grid()

pl.savefig('times.eps', format='eps', dpi=600)

pl.show()

