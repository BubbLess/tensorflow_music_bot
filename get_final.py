import midi
pattern = midi.read_midifile('out_0.mid')
for x in range(49):
    x=x+1
    try:
        q=midi.read_midifile('out_'+str(x)+'.mid')
    except:
        continue
    del pattern[0][len(pattern[0])-1]
    for tick in q[0]:
        pattern[0].append(tick)

midi.write_midifile('out_final.mid',pattern)