def create_axis(coords=[430, 200, 110], dist=20, filename='axis.bild'):
    '''Generate axes arrows in chimeraX'''
    lines = ['.color green',
             '.dot {0} {1} {2}'.format(*coords),
             '.color white',
             '.arrow {0} {1} {2} {3} {1} {2} 1 4 .6'.format(
                 *coords, coords[0]-dist),
             '.color green',
             '.dot {0} {1} {2}'.format(*coords),
             '.color red',
             '.arrow {0} {1} {2} {0} {3} {2} 1 4 .6'.format(
                 *coords, coords[1]+dist),
             '.color green',
             '.dot {0} {1} {2}'.format(*coords),
             '.color blue',
             '.arrow {0} {1} {2} {0} {1} {3} 1 4 .6'.format(*coords, coords[2]+dist)]

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def main(isChimerax=False):
    import argparse
    parser = argparse.ArgumentParser(
        description='Create axes arrow files for ChimeraX.')
    parser.add_argument("x", nargs='?', type=int,
                        default=400, help="x: 400")
    parser.add_argument("y", nargs='?', type=int,
                        default=200, help="y: 200")
    parser.add_argument("z", nargs='?', type=int,
                        default=100, help="z: 100")
    parser.add_argument("d", nargs='?', type=int,
                        default=20, help="length: 20")
    parser.add_argument("--f", dest="filename",nargs='?', type=str, default="axis.bild",
                        help="axis.bild")
    args = parser.parse_args()
    create_axis([args.x, args.y, args.z], args.d, args.filename)
    if isChimerax:
        from chimerax.core.commands import run
        run(session,'open {}'.format(args.filename))
	
if __name__ == '__main__':
    main()
elif __name__.startswith('ChimeraX'):
    main(isChimerax=True)
	

